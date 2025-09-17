#!/bin/bash
# Set GitHub username
git config --global user.name "aliciajocelyn"


# Set GitHub email (optional but recommended)
git config --global user.email "alisiahaya.college@gmail.com"

# Store credentials (username and PAT) in git credential helper
git config --global credential.helper store


# Create a file to store the credentials


# Make sure the .git-credentials file is properly protected
chmod 600 ~/.git-credentials

# Confirm credentials are stored
echo "GitHub credentials set for user aliciajocelyn"