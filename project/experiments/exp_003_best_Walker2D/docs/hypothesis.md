# What is tested?
Does using SAC + gSDE + StackFrame4 better than SAC + gSDE reported in gSDE paper?

# What are Treatment and Control groups?
Control:
SAC+gSDE from gSDE paper. Expected return 2270 +/- 28

Treatment:
SAC + gSDE + StackFrame4
SAC + gSDE + StackFrame4 + SkipFrame

# What are the results? (in short)

slightly better. At around 6M steps (with 2 std):

control
2038.779 +- 290.108
treatment: stackframe
2158.110 +- 227.775

# Other Thoughts?

Forget about Skip.

But for Stack Frame 4, there seems to be significant improvement! exciting!
