#!/usr/bin/env python3
"""
Diagnose why a Python process is hanging
"""
import asyncio
import gc
import sys
import threading
import traceback

import psutil


def diagnose_process(pid: int):
    """Comprehensively diagnose a hanging process"""
    try:
        p = psutil.Process(pid)
        print(f"\n{'='*80}")
        print(f"PROCESS DIAGNOSTICS FOR PID {pid}")
        print(f"{'='*80}\n")

        # Basic info
        print(f"Status: {p.status()}")
        print(f"Name: {p.name()}")
        print(f"Cmdline: {' '.join(p.cmdline())}")
        print(f"CWD: {p.cwd()}")
        print(f"Started: {p.create_time()}")

        # Threads
        print(f"\n{'='*80}")
        print(f"THREADS ({len(p.threads())} total)")
        print(f"{'='*80}")
        for i, thread in enumerate(p.threads(), 1):
            print(
                f"{i}. Thread ID: {thread.id}, User time: {thread.user_time:.2f}s, System time: {thread.system_time:.2f}s"
            )

        # Open files
        print(f"\n{'='*80}")
        print(f"OPEN FILES ({len(p.open_files())} total)")
        print(f"{'='*80}")
        for i, f in enumerate(p.open_files(), 1):
            if i > 20:
                print(f"... and {len(p.open_files()) - 20} more")
                break
            print(f"{i}. {f.path} (fd={f.fd})")

        # Network connections
        print(f"\n{'='*80}")
        print(f"NETWORK CONNECTIONS ({len(p.connections())} total)")
        print(f"{'='*80}")
        for i, conn in enumerate(p.connections(), 1):
            if i > 10:
                print(f"... and {len(p.connections()) - 10} more")
                break
            print(f"{i}. {conn}")

        # Child processes
        children = p.children(recursive=True)
        print(f"\n{'='*80}")
        print(f"CHILD PROCESSES ({len(children)} total)")
        print(f"{'='*80}")
        for child in children:
            print(f"PID {child.pid}: {child.name()} - {child.status()}")

        # Memory
        mem = p.memory_info()
        print(f"\n{'='*80}")
        print(f"MEMORY USAGE")
        print(f"{'='*80}")
        print(f"RSS: {mem.rss / 1024 / 1024:.1f} MB")
        print(f"VMS: {mem.vms / 1024 / 1024:.1f} MB")

        # CPU
        print(f"\n{'='*80}")
        print(f"CPU USAGE")
        print(f"{'='*80}")
        print(f"CPU %: {p.cpu_percent(interval=1.0):.1f}%")

    except psutil.NoSuchProcess:
        print(f"Process {pid} not found")
        return
    except Exception as e:
        print(f"Error analyzing process: {e}")
        import traceback

        traceback.print_exc()


def inject_thread_dump(pid: int):
    """
    Try to inject into the process and get Python thread info
    This only works if we're analyzing our own process
    """
    if pid != psutil.Process().pid:
        print("\nCannot inject into external process - skipping Python-level diagnostics")
        return

    print(f"\n{'='*80}")
    print(f"PYTHON THREAD DUMP (INTERNAL)")
    print(f"{'='*80}")

    for thread_id, frame in sys._current_frames().items():
        print(f"\n--- Thread {thread_id} ---")
        traceback.print_stack(frame)

    # Check for running threads
    print(f"\n{'='*80}")
    print(f"THREADING MODULE THREADS")
    print(f"{'='*80}")
    for thread in threading.enumerate():
        print(f"Thread: {thread.name}")
        print(f"  Daemon: {thread.daemon}")
        print(f"  Alive: {thread.is_alive()}")
        print(f"  Ident: {thread.ident}")

    # Check asyncio
    print(f"\n{'='*80}")
    print(f"ASYNCIO EVENT LOOPS")
    print(f"{'='*80}")
    try:
        loop = asyncio.get_event_loop()
        print(f"Event loop: {loop}")
        print(f"Running: {loop.is_running()}")
        print(f"Closed: {loop.is_closed()}")

        if not loop.is_closed():
            tasks = asyncio.all_tasks(loop)
            print(f"\nPending tasks: {len(tasks)}")
            for task in tasks:
                print(f"  - {task.get_name()}: {task}")
    except RuntimeError as e:
        print(f"No event loop: {e}")

    # Check for circular references
    print(f"\n{'='*80}")
    print(f"GARBAGE COLLECTOR INFO")
    print(f"{'='*80}")
    print(f"Garbage objects: {len(gc.garbage)}")
    print(f"Tracked objects: {len(gc.get_objects())}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_hang.py <PID>")
        sys.exit(1)

    pid = int(sys.argv[1])
    diagnose_process(pid)

    # If it's our own process, do deeper analysis
    if pid == psutil.Process().pid:
        inject_thread_dump(pid)
