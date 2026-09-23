# Cloudflare Tunnel and Access

 _These instructions were created in September 2026_
 
This guide configures Cloudflare Tunnel and Cloudflare Access for the DevBuilder Gradio UI. Cloudflare Tunnel connects the public hostname to the UI running on the DigitalOcean Droplet, while Cloudflare Access restricts who can open it.

The tunnel is remotely managed through the Cloudflare dashboard. The `cloudflared` connector runs as part of the VPS Docker Compose stack and establishes an outbound connection to Cloudflare. The Droplet therefore does not need public HTTP or HTTPS ports, and only the Gradio UI is exposed through the hostname. FastAPI and Qdrant remain internal services.

For general deployment prerequisites and the complete application workflow, see the [project README](../README.md). For background on tunnels, see the [Cloudflare Tunnel overview](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/).

## Prerequisites

Before beginning, you need:
- A domain managed through Cloudflare.
- One-time PIN configured as an identity provider (see below).
- A planned public hostname, such as `devtest.example.com`.
- The DevBuilder repository configured locally.
- A DigitalOcean Droplet on which to run the CPU-side stack.

The Runpod GPU Pod is not required to configure or verify Cloudflare Tunnel and Access. Without the Pod, the Gradio UI can load, but generation requests will not complete.

## 1. Configure One-time PIN

This project uses Cloudflare's [One-time PIN login](https://developers.cloudflare.com/cloudflare-one/integrations/identity-providers/one-time-pin/), which sends a temporary authentication code by email.

One-time PIN is configured once for the Cloudflare Zero Trust organization and can be added to any application: 

1. In the Cloudflare dashboard, go to **Zero Trust → Integrations → Identity providers**.
2. Under **Your identity providers**, select **Add new identity provider**.
3. Select **One-time PIN** and save it.

If One-time PIN is already listed under the organization's identity providers, do not add it again.

## 2. Create the Access application

Create the Access application before publishing the tunnel route. This prevents the hostname from being temporarily available without authentication. For additional information, see Cloudflare's [self-hosted application instructions](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/self-hosted-public-app/). Start on the main dashboard.

1. Go to **Zero Trust → Access controls → Applications**.
2. Select **Create new application**.
3. Under **Private destinations**, select **Public DNS**.
4. Continue with **Self-hosted and private**.
5. Select **Add public hostname**.
6. Enter the subdomain and domain planned for the Gradio UI. Leave the path blank.
7. Give the application a name, such as `devtest`.
8. Under **Access policies**, select **Create new policy** and configure:

   ```text
   Policy name: allow-test-user
   Action: Allow
   Include: Emails
   Value: <permitted-email-address>
   ```

9. Under **Authentication**, choose the identity providers available for the application and select **onetimepin** from the drop down menu..
10. Create/Save the application.

## 3. Create the tunnel

1. In the Cloudflare dashboard, go to **Networking → Tunnels**.
2. Select **Create a tunnel**.
3. Enter a tunnel name, such as `devtest`.
4. Select **Docker** as the connector environment.
5. Cloudflare displays a Docker command containing `--token <TUNNEL_TOKEN>`.
6. Copy only the token from that command. Do not run the generic Docker command because DevBuilder starts `cloudflared` through Docker Compose.
7. Add the token to the local `.env` file:

   ```dotenv
   TUNNEL_TOKEN=<cloudflare-tunnel-token>
   ```
The dashboard will wait for a connector before enabling **Continue**. Leave this page open while starting the VPS stack.


## 4. Start the connector on the Droplet

If the Droplet and CPU-side services are not already running, deploy them using the normal project workflow.

Run the provisioning and synchronization commands from the local machine:

```bash
make droplet-up
make droplet-sync-config
make droplet-sync-data
make remote-ghcr-login  # required only for private GHCR packages
make droplet-ssh
```

On the Droplet:

```bash
cd /opt/devbuilder
make vps-export
make vps-indexer
make vps-up
exit
```

Back on the local machine:

```bash
make remote-ghcr-logout
```

`make vps-up` starts Qdrant, FastAPI, Gradio, and `cloudflared`. 

## 5. Add the published application route

Once the connector authenticates with `TUNNEL_TOKEN`, the **Continue** button becomes available. 

1. Select **Continue** (if the tunnel-creation screen is still open).
2. Select **Add route**.
3. Choose **Published application**.
4. Configure the route:

   ```text
   Subdomain: devtest
   Domain: <your-domain>
   Path: leave blank
   Service type: HTTP
   Service URL: http://ui:7860
   ```
5. Save the route.

## 6. Verify the complete path

1. Confirm that the tunnel shows **Healthy** under **Networking → Tunnels**.
2. Open a private or incognito browser window so an existing Access session is not reused.
3. Visit the public hostname, should be of the form `devtest.example.com`.
4. Confirm that Cloudflare Access asks for an email address.
5. Enter an email address included in the Allow policy.
6. Enter the one-time PIN sent by Cloudflare.
7. Confirm that the Gradio UI loads after authentication.


## Troubleshooting

### Tunnel remains Inactive

The `cloudflared` connector is not connected. Confirm that the VPS stack is running and inspect the connector logs:

```bash
docker compose -f compose.yaml -f compose.vps.yaml --profile vps logs cloudflared
```

Also confirm that the current tunnel token was synchronized to the Droplet.

### The hostname opens without an Access challenge

Confirm that:

- The Access application exists.
- Its public hostname exactly matches the tunnel route.
- The policy action is **Allow**, with the intended email addresses under **Include → Emails**.
- There is no **Bypass** policy.
- The browser is not reusing an existing Access session.

### Access succeeds, followed by `502 Bad Gateway`

Cloudflare can reach the tunnel, but `cloudflared` cannot reach Gradio. Confirm that the UI container is running and that the published application route uses:

```text
http://ui:7860
```

### No OTP email arrives

Confirm that:

- One-time PIN is configured under **Zero Trust → Integrations → Identity providers**.
- One-time PIN is selected under the application's **Authentication** settings.
- The submitted email address is included in the application's Allow policy.
- The email provider is not blocking messages from Cloudflare.

### `.env` reports `command not found`

The tunnel token may have been placed on a separate line. It must be a single assignment:

```dotenv
TUNNEL_TOKEN=<cloudflare-tunnel-token>
```

After correcting the local `.env`, run `make droplet-sync-config` again before continuing on the Droplet.

## Resource lifecycle

The tunnel, published route, and Access application remain configured in Cloudflare when the Droplet is destroyed. A later Droplet can reconnect to the same tunnel using its connector token. The tunnel will appear **Inactive** whenever no connector is running.



