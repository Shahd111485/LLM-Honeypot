import socket
import threading
import time
import sys
import os
import paramiko
import traceback
import asyncio
import logging

# Logging system (Assuming you have this file, otherwise remove imports)
try:
    from logger import log_auth, log_cmd
except ImportError:
    # Fallback logger if file is missing
    def log_auth(u, p): print(f"[AUTH] {u}:{p}")
    def log_cmd(c, r): print(f"[CMD] {c} -> {r[:20]}...")

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SSH_Server")

# Log Paramiko for debugging (Be careful, this file grows fast)
paramiko.util.log_to_file("paramiko.log")

# Load/create RSA host key
HOST_KEY_PATH = 'test_rsa.key'
if not os.path.exists(HOST_KEY_PATH):
    logger.info(f"Generating new host key at {HOST_KEY_PATH}...")
    key = paramiko.RSAKey.generate(2048)
    key.write_private_key_file(HOST_KEY_PATH)

HOST_KEY = paramiko.RSAKey(filename=HOST_KEY_PATH)
logger.info(f"Loaded host key from {HOST_KEY_PATH}")


# =====================================================
#              SSH SERVER INTERFACE
# =====================================================
class HoneyPotInterface(paramiko.ServerInterface):
    def __init__(self):
        self.event = threading.Event()

    def check_channel_request(self, kind, chanid):
        if kind == 'session':
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_auth_password(self, username, password):
        log_auth(username, password)
        return paramiko.AUTH_SUCCESSFUL

    def check_channel_shell_request(self, channel):
        self.event.set()
        return True


# =====================================================
#          SAFE ASYNC LLM EXECUTION
# =====================================================
async def process_command(llm_instance, cmd, history):
    """
    Wraps the LLM call to ensure it never crashes the SSH thread.
    """
    try:
        # The new LLM.answer() is async, so we await it here
        response = await llm_instance.answer(cmd, history)
        return str(response)
    except Exception as e:
        logger.error(f"LLM Bridge Error: {e}")
        return "bash: command not found" # Fail silently to look like real Linux


# =====================================================
#           CONNECTION HANDLER
# =====================================================
def handle_connection(client_sock, llm_instance):
    transport = None
    loop = None
    try:
        transport = paramiko.Transport(client_sock)
        transport.add_server_key(HOST_KEY)
        server = HoneyPotInterface()

        try:
            transport.start_server(server=server)
        except paramiko.SSHException:
            logger.warning("SSH negotiation failed")
            return

        chan = transport.accept(20)
        if chan is None:
            logger.warning("No channel opened (Timeout/Auth failure)")
            return

        server.event.wait(10)
        chan.send("Welcome to Ubuntu 22.04.2 LTS\r\n\r\n")

        prompt = "root@server:~# "
        history = []
        buff = ""

        chan.send(prompt)

        # ==========================================
        # 1. SETUP ASYNC BRIDGE
        # ==========================================
        # Create a new event loop strictly for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            byte = chan.recv(1)
            if not byte:
                break

            char = byte.decode("utf-8", errors="ignore")

            # ENTER PRESSED
            if char in ("\r","\n"):
                chan.send("\r\n")
                cmd = buff.strip()

                if cmd:
                    if cmd == "exit":
                        break

                    # ==========================================
                    # 2. EXECUTE ASYNC TASK SYNCHRONOUSLY
                    # ==========================================
                    # This bridges the gap between Paramiko (Sync) and LLM (Async)
                    response = loop.run_until_complete(
                        process_command(llm_instance, cmd, history)
                    )

                    log_cmd(cmd, response)

                    # Normalize line endings for SSH terminal
                    formatted = response.replace("\n", "\r\n")
                    if not formatted.endswith("\r\n"):
                        formatted += "\r\n"

                    chan.send(formatted)

                    # Update Context
                    history.append(cmd)
                    history.append(response)

                buff = ""
                chan.send(prompt)
                #ignore second char if it's \n after \r
                continue

            # BACKSPACE HANDLING
            elif char in ("\x7f", "\x08"):
                if buff:
                    buff = buff[:-1]
                    chan.send("\x08 \x08")

            # NORMAL TYPING (No Echo)
            else:
                buff += char
                

    except Exception as e:
        logger.error(f"Connection Error: {e}")
        traceback.print_exc()

    finally:
        # ==========================================
        # 3. CLEANUP RESOURCES
        # ==========================================
        if loop:
            loop.close()  # Prevents memory leaks
        if transport:
            transport.close()
        client_sock.close()
        logger.info("Connection closed")


# =====================================================
#               SERVER STARTER
# =====================================================
def start_ssh_server(llm_instance, port=2222):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        sock.bind(("0.0.0.0", port))
    except PermissionError:
        logger.error(f"Permission denied binding port {port}. Try sudo or port > 1024.")
        return

    sock.listen(100)
    logger.info(f"SSH Honeypot active on port {port}")
    logger.info(f"LLM Model: {llm_instance.api_model}")

    while True:
        try:
            client, addr = sock.accept()
            logger.info(f"Incoming connection from {addr[0]}")
            t = threading.Thread(target=handle_connection, args=(client, llm_instance))
            t.daemon = True # Kills thread if main program exits
            t.start()
        except KeyboardInterrupt:
            logger.info("Server stopping...")
            break
        except Exception as e:
            logger.error(f"Accept Error: {e}")