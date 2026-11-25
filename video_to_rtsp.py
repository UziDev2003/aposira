#!/usr/bin/env python3
"""Loop a video as an RTSP camera for Store Vision. Requires Python 3.9+ and FFmpeg."""
import argparse
import hashlib
import io
import ipaddress
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile


MEDIAMTX_VERSION = '1.21.4'
RELEASE_URL = f'https://github.com/bluenviron/mediamtx/releases/download/v{MEDIAMTX_VERSION}'


def download(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'APOSIRA-RTSP-Demo'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def find_server():
    system = platform.system().lower()
    machine = platform.machine().lower()
    architectures = {'x86_64': 'amd64', 'amd64': 'amd64', 'aarch64': 'arm64',
                     'arm64': 'arm64', 'armv7l': 'armv7', 'armv6l': 'armv6'}
    architecture = architectures.get(machine)
    if system not in ('linux', 'windows', 'darwin') or architecture is None:
        raise RuntimeError('This platform has no supported bundled MediaMTX release.')
    cache = Path(tempfile.gettempdir()) / 'aposira-rtsp-tools' / f'{MEDIAMTX_VERSION}-{system}-{architecture}'
    binary_name = 'mediamtx.exe' if system == 'windows' else 'mediamtx'
    binary = cache / binary_name
    if binary.is_file():
        return str(binary)
    extension = 'zip' if system == 'windows' else 'tar.gz'
    filename = f'mediamtx_v{MEDIAMTX_VERSION}_{system}_{architecture}.{extension}'
    print(f'Downloading MediaMTX {MEDIAMTX_VERSION} from its official GitHub release…', flush=True)
    checksums = download(f'{RELEASE_URL}/checksums.sha256').decode()
    expected = next((line.split()[0] for line in checksums.splitlines()
                     if line.split()[-1].lstrip('*') == filename), None)
    if expected is None:
        raise RuntimeError(f'The official release has no checksum for {filename}.')
    archive = download(f'{RELEASE_URL}/{filename}')
    if hashlib.sha256(archive).hexdigest() != expected:
        raise RuntimeError('MediaMTX download checksum did not match. Run the script again.')
    if extension == 'zip':
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            executable = bundle.read(binary_name)
    else:
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as bundle:
            member = bundle.extractfile(binary_name)
            if member is None:
                raise RuntimeError('The release archive does not contain MediaMTX.')
            with member:
                executable = member.read()
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temporary:
        temporary.write(executable)
        temporary_name = temporary.name
    try:
        os.chmod(temporary_name, 0o755)
        os.replace(temporary_name, binary)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return str(binary)


def start_process(command, name, log_directory):
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, text=True, encoding='utf-8',
                               errors='replace', bufsize=1, creationflags=flags,
                               cwd=log_directory)

    def show_output():
        with process.stdout, (log_directory / f'{name}.log').open('w', encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f'[{name}] {line.rstrip()}', flush=True)

    threading.Thread(target=show_output, daemon=True).start()
    return process


def stream_available(host, port, path):
    try:
        with socket.create_connection((host, port), timeout=1) as connection:
            request = (f'DESCRIBE rtsp://{host}:{port}/{path} RTSP/1.0\r\n'
                       'CSeq: 1\r\nAccept: application/sdp\r\n\r\n')
            connection.sendall(request.encode())
            return connection.recv(4096).startswith(b'RTSP/1.0 200 ')
    except OSError:
        return False


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run(args):
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise ValueError(f'Video file does not exist: {video}')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('FFmpeg is required. On RunPod run: apt-get update && apt-get install -y ffmpeg')
    # Fail before starting children if another service already owns the port.
    with socket.socket() as check:
        check.bind((args.bind, args.port))
    server_binary = find_server()
    logs = Path(tempfile.mkdtemp(prefix='aposira-rtsp-'))
    configuration = logs / 'mediamtx.yml'
    configuration.write_text(
        f'logLevel: warn\nrtsp: true\nrtspAddress: {args.bind}:{args.port}\n'
        'rtspTransports: [tcp]\nrtmp: false\nhls: false\nwebrtc: false\nsrt: false\nmoq: false\n'
        f'paths:\n  "{args.path}":\n    source: publisher\n', encoding='utf-8')
    local_host = '127.0.0.1' if args.bind == '0.0.0.0' else args.bind
    url = f'rtsp://{local_host}:{args.port}/{args.path}'
    server = publisher = None
    try:
        print(f'Logs: {logs}', flush=True)
        server = start_process([server_binary, str(configuration)], 'server', logs)
        deadline = time.monotonic() + 10
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'MediaMTX exited with code {server.returncode}. See {logs}.')
            try:
                with socket.create_connection((local_host, args.port), timeout=0.5):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'MediaMTX did not open port {args.port}. See {logs}.')
                time.sleep(0.2)
        publisher = start_process([
            ffmpeg, '-hide_banner', '-loglevel', 'warning', '-nostdin',
            '-re', '-stream_loop', '-1', '-i', str(video), '-map', '0:v:0', '-an',
            '-vf', f'fps={args.fps},scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p', '-g', str(args.fps * 2), '-bf', '0',
            '-f', 'rtsp', '-rtsp_transport', 'tcp', url,
        ], 'publisher', logs)
        deadline = time.monotonic() + 30
        while not stream_available(local_host, args.port, args.path):
            if publisher.poll() is not None or server.poll() is not None:
                raise RuntimeError(f'The stream stopped during startup. See {logs}.')
            if time.monotonic() >= deadline:
                raise RuntimeError(f'No readable RTSP video was published within 30 seconds. See {logs}.')
            time.sleep(0.3)
        print(f'\nRTSP stream ready: {url}\n'
              'Store Vision → Live Stream (Webcam/RTSP) → RTSP → paste this URL → Start.\n'
              'The video loops continuously. Keep this script running. Ctrl+C stops it.\n', flush=True)
        if args.bind == '0.0.0.0':
            print(f'Remote viewers use rtsp://<server-IP>:<mapped-TCP-port>/{args.path}', flush=True)
        started = time.monotonic()
        while not args.duration or time.monotonic() - started < args.duration:
            if server.poll() is not None or publisher.poll() is not None:
                raise RuntimeError(f'The RTSP server or publisher stopped. See {logs}.')
            time.sleep(0.5)
    finally:
        stop_process(publisher)
        stop_process(server)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path, help='MP4, MOV, AVI or another FFmpeg-readable video')
    parser.add_argument('--port', type=int, default=8554)
    parser.add_argument('--path', default='store', help='RTSP stream name (default: store)')
    parser.add_argument('--bind', default='127.0.0.1', help='Listener IPv4 address; use 0.0.0.0 for remote readers')
    parser.add_argument('--fps', type=int, default=15, help='Output frames per second, at normal playback speed')
    parser.add_argument('--duration', type=float, default=0, help='Stop after this many streaming seconds; 0 loops indefinitely')
    args = parser.parse_args()
    if (not 1 <= args.port <= 65535 or not 1 <= args.fps <= 60
            or not math.isfinite(args.duration) or args.duration < 0):
        parser.error('Use port 1–65535, FPS 1–60, and a nonnegative duration.')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.path):
        parser.error('Stream path must contain only letters, numbers, hyphens or underscores.')
    try:
        ipaddress.IPv4Address(args.bind)
    except ipaddress.AddressValueError:
        parser.error('--bind must be an IPv4 address.')

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(args)
    except KeyboardInterrupt:
        print('\nRTSP stream stopped.', flush=True)
    except (OSError, ValueError, RuntimeError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
