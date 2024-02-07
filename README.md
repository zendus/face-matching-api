# Passport Photo Face Detection and Matching API

This API is built using FastAPI and is designed to take a passport photograph as input, utilize OpenCV and dlib with HOG and CNN models to detect faces, extract face features and descriptors, and calculate the Euclidean distance to determine the difference between the input face and other faces residing in a PostgreSQL identity database. It returns the distance value as a matching score along with a count of successful face matches.

## Prerequisites

Before running this API, ensure you have the following installed:

- Python 3.x
- FastAPI
- OpenCV
- dlib
- PostgreSQL

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/zendus/face-matching-api.git
   ```

2. Install the required Python packages:

    ```bash
    pipenv install
   pip install pipenv
   ```

3. Set up your PostgreSQL database and configure the connection in `database.py`.

## Usage

1. Run the FastAPI server:

   ```bash
   uvicorn app.main:app --reload
   ```

2. Make a POST request to `http://localhost:8000/match-passport` with a passport photograph as input in the request body.

3. Receive the matching score, id of matched user and count of successful face matches as the response.

## API Endpoints

- `/match-passport`: Accepts a passport photograph as input and returns the matching score, id of matched user and count of successful face matches.

- `/submit-passport`: Accepts a passport photograph as input and returns a json containing user id and passport cloudinary url.

## Example

```bash
curl -X POST -F "photo=@passport_photo.jpg" http://localhost:8000/match-passport
```

## Documentation 

You can go to the link `http://localhost:8000/docs` to access the API swagger documentation.

## Acknowledgments

This API utilizes OpenCV and dlib libraries for face detection and feature extraction. It also makes use of PostgreSQL for identity database management.

## License

This project has no license.

## Contributors

- [Johnmicheal Uzendu](https://github.com/zendus)

## Contact

For any inquiries, please contact [j.uzendu@icloud.com](mailto:j.uzendu@icloud.com).