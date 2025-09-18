Currently i am having issues with convergence. The optic begins moving towards something
sensible, then will slowly diverge to a messy, high frequency optic with lots of weird behaviour and 
relatively high loss.

Tests:

Keeping the input/output training set the same for ~100 iterations before changing
The idea here is that we can move real rays to where they ought to go, rather than
boiling back and forth for different rays. This seems to smooth out the training curve
a lot (the noise is caused by the input / outputs changing).

Keeping the loss function constant (removing schedule based loss change) is also
something that I theorize might help: changing the loss function changes what we're
even optimizing for. It might be possible that the optimal solution is not smoothly
varying with the schedule, and we're getting stuck against optimization barriers?
I didn't notice any big change from this yet.

I should clean up the code so I can do tests like this a little more easily... And also
record results / plots / volumes alongside tests.

Ideas from jordao:
- ramp up the learning rate so that the initial batches can only make small changes
- use a cyclic LR to avoid local minima
