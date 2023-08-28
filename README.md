# Trading Core ML (backend)

## Owner information
- Group name: H2A
- University: University of Science - VNUHCM

| Student Name    | Student ID | Email                       |
| --------------- | ---------- | --------------------------- |
| Huy Le Minh     | 19127157   | leminhhuy.hcmus@gmail.com   |
| Anh Hoang Le    | 19127329   | lehoanganh.le2001@gmail.com |
| Hung Nguyen Hua | 19127150   | huahung.nguyen01@gmail.com  |


## Tech-Stack
- Programming language: python (python 3.9.*)
- Rest API: [FastAPI framework](https://fastapi.tiangolo.com/)
- Websocket: [python-socketio](https://python-socketio.readthedocs.io/en/latest/)
- Data source:
    - [Binance Rest API](https://binance-docs.github.io/apidocs/spot/en/#introduction)
    - [Binance Websocket Stream](https://github.com/LUCIT-Systems-and-Development/unicorn-binance-websocket-api)
- Machine learning models: XGBoost, RNN, LSTM

## Virtual Environment
### 1. Purposes
- All dependencies have installed in isolation environment (same as `node_modules` for NodeJS).

### 2. Setup
- Install `virtualenv` package:
    ```sh
    pip install virtualenv
    ```
- Create environment at the root directory (syntax below):
    ```sh
    python3 -m venv venv
    ```
- Activate environment:
    - MacOS:
        ```sh
        source venv>/bin/activate
        ```
    - Windows:
        ```sh
        # command line
        vevn\Scripts\activate.bat

        # power shell
        vevn\Scripts\Activate.ps1
        ```
- Deactivate environment:
    ```sh
    deactivate
    ```

### 3. Usages
- You must activate virutal environment before installing pip dependencies.
- Install neccessary dependencies for this project:
    - Install dependencies in `requirements.txt` files:
        ```
        pip install -r requirements.txt
        ```
    - Install `tensorflow 2.13.0`
        ```
        pip install tensorflow==2.13.0
        ```
    - Fix conclict about versoning of `typing_extensions` dependency between tensorflow and others. Execute the below command after installing `tensorflow`:
        ```
        pip install typing_extensions==4.7.1
        ```

- After activating your virtual environment, you could install any dependencies using `pip`:
    ```
    pip install <dependency-name>
    ```

## Instruction to start app
- You must do setup virtual environment and install pip dependencies as above.
- Steps to run app:
    - Activate virtual environment as instruction above:
    - Move to `src` directory and run the following command to run app:
        ```sh
        python main.py
        ```
    - App will be running on <SERVER_HOST>:<SERVER_PORT> depending on your `.env` file

## References
- [Project template](https://github.com/Aeternalis-Ingenium/FastAPI-Backend-Template/tree/trunk/backend/src)
- [Python SocketIO](https://python-socketio.readthedocs.io/en/latest/server.html)
