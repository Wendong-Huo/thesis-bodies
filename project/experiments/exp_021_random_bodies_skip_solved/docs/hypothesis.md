# What is tested?

Enable self collision for random bodies. (More realistic.)
Need to re-search for workable bodies.

Generate 100 bodies, train them with 2 runs and pick the best 8 bodies.
Name them 100s.

Train on the best 8 bodies simultaneously, with different alignments (100 alignments) (each with 3 runs). 300 jobs.

# What are Treatment and Control groups?


# What are the results? (in short)


# Other Thoughts?

I really should employ GA, otherwise I can't find the good alignment for these bodies!


20 workable bodies are in `input_data/bodies`.
The trained models for them are in `output_data/models`.
