import subprocess
import tkinter as tk
from tkinter import messagebox

def run_script(script_name):
    subprocess.run(["python", script_name], check=True)

def popup_message(title, message, kind="info"):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.lift()
    root.after(0, root.focus_force)

    if kind == "info":
        messagebox.showinfo(title, message, parent=root)
    elif kind == "yesno":
        return messagebox.askyesno(title, message, parent=root)

    root.destroy()

def confirm_long_process():
    return popup_message(
        "Confirm",
        "Regenerating sentiment scores can take up to an hour.\nProceed?",
        kind="yesno"
    )


def notify_complete():
    popup_message("Done", "Sentiment generation has finished.")

def main():
    while True:
        print("\nSelect an option:")
        print("1. Re-run sentiment generation")
        print("2. View results")
        print("3. Exit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            confirmed = confirm_long_process()
            if not confirmed:
                continue

            try:
                run_script("qual_manual.py")
                run_script("qual_vectorsnew.py")
                run_script("qual_llm_reading.py")
            except Exception as e:
                print(f"Error during generation: {e}")
                continue

            notify_complete()

        elif choice == "2":
            try:
                run_script("qual_predict_manual.py")
                run_script("qual_predict_vector.py")
                run_script("qual_predict_llm.py")
            except Exception as e:
                print(f"Error while viewing results: {e}")
                continue

        elif choice == "3":
            break

        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
