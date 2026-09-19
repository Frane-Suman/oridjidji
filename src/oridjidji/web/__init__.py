"""The web layer (brandbook). Templates, static assets and view-model construction.

Kept out of `api/` because the HTTP contract and the interface that consumes it are
separate concerns: the contract is what M4-M7 fill in, the interface is what a person
reads. `api/app.py` mounts this.
"""
