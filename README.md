# OpenAwareness

OpenAwareness is a language model that has undergone extensive training on over **1000** back-and-forth conversations, accumulating a total training time of over **5** hours!

This rigorous training process has equipped OpenAwareness with an understanding of the English language, enabling it to engage in general conversations.

You can also train your own model using the OpenAwareness platform, which provides tools and resources for training custom language models tailored to specific domains or use cases.

## API Endpoints

**`GET /train`**
- Description: Trains the model.
- Output: `JSON`
- Method: `GET`
- Parameters:
    - `data (str)`: Training data parameter.
    - `epochs (int)`: Number of epochs parameter.

**`GET /generate`**
- Description: Generates output using the model.
- Method: `GET`
- Output: `JSON`
- Parameters:
  - `seed (str)`: Prompt parameter.
  - `max (int)`: *OPTIONAL* Maximum number of words parameter.

**`GET /load`**
- Description: Loads the pretrained `OpenAwareness` model.
- Method: `GET`
- Output: `JSON`

**`GET /reset`**
- Description: Resets the model.
- Method: `GET`
- Output: `JSON`
