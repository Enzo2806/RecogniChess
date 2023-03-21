# RecogniChess

A program that uses Domain Adaptation to provide automatic annotations of chess games inputed in image form.

# How to use virtual environments the first time

## Step 1 - Create the virtual environment

Navigate to the RecogniChess folder and run the command "python -m venv myenv".

## Step 2 - Activate your venv

If using VSCode, download the venv extension and activate the virtual environment, then launch a terminal from within that venv.

## Step 3 - Install the dependencies from the requirements.txt file

From within the venv terminal, run "pip install -r requirements.txt" to install the dependencies used up until that point in the project.

# Recurrent use of the virtual environment

## Activate the virtual environment

See step 2 above.

## Adding new dependencies

From within a venv terminal, run "pip install <dependency>". Don't forget to then run "pip freeze > requirements.txt" to update the requirements file for others to use.
