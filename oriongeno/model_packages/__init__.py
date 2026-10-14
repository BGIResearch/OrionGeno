"""Packaged assets with prediction classes loaded only when requested."""

__all__ = ["OrionGenoConfig", "OrionGenoForJointLabeling"]


def __getattr__(name):
    if name in __all__:
        from . import orion_geno

        value = getattr(orion_geno, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
