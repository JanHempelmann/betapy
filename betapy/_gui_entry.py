def main():
    try:
        from betapy.gui.app import main as _main
    except ImportError:
        print(
            "betapy GUI dependencies are not installed.\n"
            "Install them with:  pip install 'betapy[gui]'"
        )
        raise SystemExit(1)
    _main()
