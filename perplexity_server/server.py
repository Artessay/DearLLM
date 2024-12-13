import sys
import json
import pickle
import socket
import datetime
import threading

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perplexity_server.model import LargeLanguageModel
from perplexity_server.scheduler import ModelPool

SERVER_LOG = "server.log"

class Server(object):
    def __init__(self, host, port, backlog=128, buffer_size=4096):
        # socket
        self.host = host
        self.port = port
        self.backlog = backlog
        self.buffer_size = buffer_size
        self.socket = None

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(self.backlog)
        print("Server listening on port {}".format(self.port))

    def accept(self):
        return self.socket.accept()

    def close(self):
        self.socket.close()
    
    def run(self):
        assert False

class PerplexityServer(Server):
    def __init__(self, host, port, model_name_or_path, n_models, text_template, ehr_data_path, backlog=128, buffer_size=4096):
        # socket
        super(PerplexityServer, self).__init__(host, port, backlog, buffer_size)

        # model pool
        self.pool = ModelPool(model_name_or_path, n_models=n_models)

        # format prompt
        self.text_template: str = text_template

    def run(self):
        try:
            while True:
                client, address = self.accept()
                
                # handle client connection
                client_thread = threading.Thread(target=self.connection_handler, args=(client, address))
                client_thread.start()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print("[ERROR]", e)

    def connection_handler(self, client: socket, address):
        # print("{} Client connected from {}:{}".format(datetime.datetime.now(), address[0], address[1]))

        # receive data from client
        try:
            serialized_data = client.recv(self.buffer_size)

            received_data = pickle.loads(serialized_data)
            (patient_id, disease1, disease2) = received_data
        except Exception as e:
            print("Error occurred while receiving data from client: {}".format(e), file=sys.stderr)
            client.close()
            return

        prompt = self.text_template.format(disease1)

        model: LargeLanguageModel = self.pool.acquire()
        try:
            # print("  Model acquired: {}".format(model))
            result = model.calculate_perplexity_for_text(prompt, disease2)
        except Exception as e:
            print(e, file=sys.stderr)
        finally:
            self.pool.release(model)
            # print("  Model released: {}".format(model))
        
        # print("  Result of calculation: {}".format(result))
        print("{} [result]: {} [patient]: {} [condition1]: {} [condition2]: {}".format(datetime.datetime.now(), result, patient_id, disease1, disease2))

        try:
            client.send(str(result).encode())
        except Exception as e:
            print("Error occurred while sending data to client: {}".format(e), file=sys.stderr)
            client.close()
            return

        client.close()

if __name__ == "__main__":
    from config import (
        PERPLEXITY_SERVER_PORT, 
        PERPLEXITY_TEXT_TEMPLATE,
        EHR_DATA_PATH,
        LLM_MODEL_PATH, 
        LLM_MODEL_NUM
    )
    
    server = PerplexityServer(
        host="0.0.0.0", 
        port=PERPLEXITY_SERVER_PORT, 
        model_name_or_path=LLM_MODEL_PATH, 
        n_models=LLM_MODEL_NUM,
        text_template=PERPLEXITY_TEXT_TEMPLATE,
        ehr_data_path=EHR_DATA_PATH
    )
    try:
        server.start()
        sys.stdout = open(SERVER_LOG, "w")
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout = sys.__stdout__
        print("Server shutting down...")
        server.close()

    