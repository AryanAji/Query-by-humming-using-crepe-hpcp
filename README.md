# CREPE+HPCP Real-Time Humming

My project combines **CREPE** (a deep pitch embedding) with **HPCP** (harmonic pitch class profile),
to perform real-time Query-by-Humming.

## Installation
1. Clone this repo
2. `pip install -r requirements.txt`
3. `python reference_extractor.py` to generate embeddings
4. `python app.py` and open http://127.0.0.1:5000

## Features
- Real-time humming detection and matching against a database of songs.
- Utilizes CREPE for pitch extraction and HPCP for harmonic profile comparison.
- Flask-based web application with Socket.IO for real-time updates.

## License
This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for details.


