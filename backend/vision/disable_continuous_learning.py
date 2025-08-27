#!/usr/bin/env python3
"""
Temporarily disable continuous learning to fix performance issues
"""


def patch_to_disable():
    """Disable continuous learning by preventing the thread from starting"""
    try:
        import vision.advanced_continuous_learning as acl

        # Override the start method to do nothing
        def disabled_start(self):
            """Disabled continuous learning to prevent CPU issues"""
            self.running = False
            self.enabled = False
            print("WARNING: Continuous learning disabled for performance")
            return

        # Apply the patch
        acl.AdvancedContinuousLearning._start_continuous_learning = disabled_start

        print("Continuous learning has been disabled")
        return True

    except Exception as e:
        print(f"Failed to disable continuous learning: {e}")
        return False


if __name__ == "__main__":
    patch_to_disable()
