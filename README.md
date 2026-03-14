# Diffusion Simulator project.
This project serves as a lighweight demonstration of how conditional diffusion models, like Stable-diffusion and the likes, work. This demonstration uses a lightweight diffusion model to generate 2D coordinate sequences by denoising 2D point clouds, based on the shape entered as a text prompt by the user.

## Features:
    1. **Conditional diffusion model**: Using text embeddings (from BERT) to guide the generation procedure.
    2. **Gradio UI**: Support for running the model interactively for people who want to see the diffusion procedure in action.

## What I got to learn.
    1. How BERT uses the [CLS] token as a complete single-token summary of the entire sentence, that's being tokenized.
    2. How to project conditions to guide the generation procedure.
    3. How to virtually increase data to make the model see more of it.
    4. How to use schedulers to lower the learning rate, anytime the loss went plateau or oscillated, instead of getting low.

## How it went.
    This project has been through many iterations in the past week.

    The major iteration being, shifting from denoising points individually to denoising the entire point-cloud of mentioned shapes in the dataset.
    So, why I did what I did?
        -> Denoising individual points made me notice that no matter the number of epochs the model was training in, the results weren't anywhere near of what one would expect. It was either just a blob of points or a scatter that made no sense.
        -> I was quick to realize the qualitative mistake of not letting the model the entire spatial representation of a shape. Thus, I made the model see the entire point cloud of a particular label.

## Future scope.
    1. Use a transformer or any other autoregressive model to remember the sequence in which the points get laid. That way, we can generate the sequence better.
    2. Use this diffusion of point clouds in path-planning for robotics.

## Bibliography
    [1]Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in neural information processing systems, 33, 6840-6851.