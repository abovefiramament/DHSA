"""Locked Site selector experiment support.

Scientific settings live in the versioned Site JSON. Modules in this package
deliberately expose artifact and device arguments only.
"""

from screscomp.site.protocol import SiteJob, SiteProtocol

__all__ = ["SiteJob", "SiteProtocol"]
