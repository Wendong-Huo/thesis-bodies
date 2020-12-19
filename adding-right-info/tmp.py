import sys,os
import pybullet as p
def makesomenoise():
    print("test")



class OutputSwitch:
    oldstdout_fno = None
    oldstderr_fno = None
    def redirect_stdout(self):
        print("Redirecting stdout")
        sys.stdout.flush() # <--- important when redirecting to files

        # Duplicate stdout (file descriptor 1)
        # to a different file descriptor number
        newstdout = os.dup(1)

        # /dev/null is used just to discard what is being printed
        devnull = os.open('/dev/null', os.O_WRONLY)

        # Duplicate the file descriptor for /dev/null
        # and overwrite the value for stdout (file descriptor 1)
        os.dup2(devnull, 1)

        # Close devnull after duplication (no longer needed)
        os.close(devnull)

        # Use the original stdout to still be able
        # to print to stdout within python
        self.oldstdout_fno = sys.stdout
        sys.stdout = os.fdopen(newstdout, 'w')
    def recover_stdout(self):
        sys.stdout = self.oldstdout_fno

    def disable_output(self):
        sys.stdout.flush() # <--- important when redirecting to files
        devnull = open('/dev/null', 'w')
        self.oldstdout_fno = os.dup(sys.stdout.fileno())
        self.oldstderr_fno = os.dup(sys.stderr.fileno())
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())
    def enable_output(self):
        if self.oldstdout_fno:
            os.dup2(self.oldstdout_fno, sys.stdout.fileno())
        if self.oldstderr_fno:
            os.dup2(self.oldstderr_fno, sys.stderr.fileno())
        

output_switch = OutputSwitch()

# output_switch.disable_output()
output_switch.redirect_stdout()
# p.connect(p.GUI)
makesomenoise()
output_switch.recover_stdout()
p.connect(p.GUI)
makesomenoise()

# output_switch.enable_output()
