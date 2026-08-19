from pathlib import Path
import os
import shutil
import signal
import subprocess
import sys
import time
import webbrowser


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = "8000"

FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = "5173"


backend_process = None
frontend_process = None


def print_header():
    print("=" * 70)
    print("SickleGuide - Full Stack Runner")
    print("=" * 70)
    print(f"Project root : {ROOT_DIR}")
    print(f"Backend      : http://{BACKEND_HOST}:{BACKEND_PORT}")
    print(f"Frontend     : http://{FRONTEND_HOST}:{FRONTEND_PORT}")
    print("=" * 70)
    print()


def check_environment():
    if not (ROOT_DIR / "api").exists():
        raise FileNotFoundError(
            f"API folder not found: {ROOT_DIR / 'api'}"
        )

    if not FRONTEND_DIR.exists():
        raise FileNotFoundError(
            f"Frontend folder not found: {FRONTEND_DIR}"
        )

    if not (FRONTEND_DIR / "package.json").exists():
        raise FileNotFoundError(
            f"Frontend package.json not found: "
            f"{FRONTEND_DIR / 'package.json'}"
        )

    python_executable = sys.executable

    npm_command = "npm.cmd" if os.name == "nt" else "npm"

    if shutil.which(npm_command) is None:
        raise RuntimeError(
            "npm was not found in PATH."
        )

    return python_executable, npm_command


def start_backend(python_executable):
    print("[1/2] Starting FastAPI backend...")

    command = [
        python_executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        BACKEND_HOST,
        "--port",
        str(BACKEND_PORT),
    ]

    process = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR),
        stdin=subprocess.DEVNULL,
    )

    print(
        f"      Backend PID: {process.pid}"
    )

    return process


def start_frontend(npm_command):
    print("[2/2] Starting React/Vite frontend...")

    command = [
        npm_command,
        "run",
        "dev",
        "--",
        "--host",
        FRONTEND_HOST,
        "--port",
        str(FRONTEND_PORT),
    ]

    process = subprocess.Popen(
        command,
        cwd=str(FRONTEND_DIR),
        stdin=subprocess.DEVNULL,
        shell=False,
    )

    print(
        f"      Frontend PID: {process.pid}"
    )

    return process


def stop_process(process, name):
    if process is None:
        return

    if process.poll() is not None:
        return

    print(f"\nStopping {name}...")

    try:
        if os.name == "nt":
            subprocess.run(
                [
                    "taskkill",
                    "/F",
                    "/T",
                    "/PID",
                    str(process.pid),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()

    except Exception as exc:
        print(
            f"Could not stop {name}: {exc}"
        )


def shutdown(*_):
    global backend_process
    global frontend_process

    print("\n")
    print("=" * 70)
    print("Shutting down SickleGuide...")
    print("=" * 70)

    stop_process(
        frontend_process,
        "frontend",
    )

    stop_process(
        backend_process,
        "backend",
    )

    print("\nSickleGuide stopped.")
    sys.exit(0)


def wait_for_servers():
    print()
    print("Starting services...")
    print()

    for _ in range(20):
        backend_alive = (
            backend_process is not None
            and backend_process.poll() is None
        )

        frontend_alive = (
            frontend_process is not None
            and frontend_process.poll() is None
        )

        if not backend_alive:
            print(
                "\nERROR: Backend process stopped."
            )
            return False

        if not frontend_alive:
            print(
                "\nERROR: Frontend process stopped."
            )
            return False

        time.sleep(0.5)

    return True


def main():
    global backend_process
    global frontend_process

    print_header()

    try:
        python_executable, npm_command = (
            check_environment()
        )

        print(
            f"Python : {python_executable}"
        )

        print(
            f"NPM    : {npm_command}"
        )

        print()

        backend_process = start_backend(
            python_executable
        )

        time.sleep(2)

        if backend_process.poll() is not None:
            raise RuntimeError(
                "Backend exited during startup."
            )

        frontend_process = start_frontend(
            npm_command
        )

        time.sleep(3)

        if frontend_process.poll() is not None:
            raise RuntimeError(
                "Frontend exited during startup."
            )

        if not wait_for_servers():
            shutdown()

        print()
        print("=" * 70)
        print("SICKLEGUIDE IS RUNNING")
        print("=" * 70)
        print()
        print(
            f"Backend API : "
            f"http://{BACKEND_HOST}:{BACKEND_PORT}"
        )
        print(
            f"Swagger     : "
            f"http://{BACKEND_HOST}:{BACKEND_PORT}/docs"
        )
        print(
            f"Frontend    : "
            f"http://{FRONTEND_HOST}:{FRONTEND_PORT}"
        )
        print()
        print(
            "Press Ctrl+C to stop both services."
        )
        print("=" * 70)

        time.sleep(1)

        try:
            webbrowser.open(
                f"http://{FRONTEND_HOST}:{FRONTEND_PORT}"
            )
        except Exception:
            pass

        while True:

            if backend_process.poll() is not None:
                print(
                    "\nBackend stopped unexpectedly."
                )
                break

            if frontend_process.poll() is not None:
                print(
                    "\nFrontend stopped unexpectedly."
                )
                break

            time.sleep(1)

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        print()
        print("=" * 70)
        print("SICKLEGUIDE STARTUP FAILED")
        print("=" * 70)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

    finally:
        shutdown()


if __name__ == "__main__":
    signal.signal(
        signal.SIGINT,
        shutdown,
    )

    main()