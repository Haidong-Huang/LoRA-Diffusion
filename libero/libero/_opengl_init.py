"""
OpenGL initialization module - must be imported before any OpenGL/robosuite imports
"""
import os
import sys

# Force set EGL platform before any OpenGL imports
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Try to initialize OpenGL platform early
try:
    # Import OpenGL platform module and force EGL initialization
    import OpenGL.platform
    import OpenGL.platform.egl as egl_platform_module
    
    # Force platform selection by directly setting the platform
    if hasattr(OpenGL.platform, 'PLATFORM'):
        try:
            # Try to create EGL platform instance
            if hasattr(egl_platform_module, 'EGLPlatform'):
                platform_instance = egl_platform_module.EGLPlatform()
                if platform_instance is not None:
                    OpenGL.platform.PLATFORM = platform_instance
        except Exception as e:
            # If EGL fails, try to use a fallback or raise a clearer error
            print(f"[WARNING] Failed to initialize EGL platform: {e}")
            print("[WARNING] You may need to install EGL libraries or use a different rendering backend")
except (ImportError, AttributeError) as e:
    # OpenGL not installed or EGL not available
    print(f"[WARNING] OpenGL EGL platform not available: {e}")
    pass
