# p_kit/utils/deprecation.py
import warnings
import functools


def warn_deprecated(
    old_name,
    new_name,
    since_version="0.1.0",
    remove_version="1.0.0",
    obj_type="function",
    example=None,
):
    """
    Generate a standardized deprecation warning for p-kit.

    Parameters
    ----------
    old_name : str
        Name of the deprecated item
    new_name : str
        Name of the replacement item
    since_version : str, optional
        Version when deprecation was introduced
    remove_version : str, optional
        Version when item will be removed
    obj_type : str, optional
        Type of object being deprecated ("function", "class", "method", etc.)
    example : str, optional
        Example code showing how to use the new approach

    Returns
    -------
    None
    """
    message = (
        f"{old_name} {obj_type} is deprecated since version {since_version} and will be "
        f"removed in version {remove_version}. Use {new_name} instead."
    )

    if example:
        message += f"\nExample:\n{example}"

    warnings.warn(message, DeprecationWarning, stacklevel=2)


def deprecated(new_name, since_version="1.0.0", remove_version="2.0.0", example=None):
    """
    Decorator to mark functions, methods or classes as deprecated.

    Parameters
    ----------
    new_name : str
        Name of the replacement item
    since_version : str, optional
        Version when deprecation was introduced
    remove_version : str, optional
        Version when item will be removed
    example : str, optional
        Example code showing how to use the new approach

    Returns
    -------
    wrapped : callable
        The wrapped function/class with deprecation warning
    """

    def decorator(obj):
        obj_type = "class" if isinstance(obj, type) else "function"

        if isinstance(obj, type):
            # If it's a class, wrap its __init__
            original_init = obj.__init__

            def wrapped_init(self, *args, **kwargs):
                warn_deprecated(
                    obj.__name__,
                    new_name,
                    since_version,
                    remove_version,
                    obj_type,
                    example,
                )
                original_init(self, *args, **kwargs)

            obj.__init__ = wrapped_init
            return obj
        else:
            # If it's a function/method
            @functools.wraps(obj)
            def wrapped(*args, **kwargs):
                warn_deprecated(
                    obj.__name__,
                    new_name,
                    since_version,
                    remove_version,
                    obj_type,
                    example,
                )
                return obj(*args, **kwargs)

            return wrapped

    return decorator
