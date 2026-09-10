"""
Infrastructure: everything that provisions, syncs, or tears down.

``cli`` is the ``devbuilder-deploy`` entry point, ``state`` the record of
what exists, ``pod`` the GPU decoder host (RunPod), ``droplet`` the CPU box
that runs the VPS stack (DigitalOcean). The application package above this
one never imports from here.
"""
