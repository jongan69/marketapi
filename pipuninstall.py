import subprocess
import sys

def uninstall_package(package_name):
    """Uninstall a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package_name])
        print(f"Successfully uninstalled {package_name}")
    except subprocess.CalledProcessError:
        print(f"Failed to uninstall {package_name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python pipuninstall.py <package_name> [package_name2 ...]")
        print("Example: python pipuninstall.py numpy pandas")
        return
    
    for package in sys.argv[1:]:
        uninstall_package(package)

if __name__ == "__main__":
    main() 