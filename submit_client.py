from shutil import copyfile
import sys
import zmq

file = sys.argv[-1]
tempfile = "/tmp/" + file.split("/")[-1]
copyfile(file, tempfile)

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect('tcp://127.0.0.1:5555')
socket.send_string(tempfile)
msg = socket.recv_string()
print(msg)
sys.stderr.write("Finished\n")
