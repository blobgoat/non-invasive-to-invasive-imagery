Author/User: the21catt

Date Finished: 8/1/2025

Goals

1.  Have git set up and have everything cloned to your laptop.
2.  install <https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images> . There is a .tcia file inside the data folder so no need to find the exact images. You just need the installer to install all the addresses within the file.
3.  Extract all images. It requires a ton of space (about 100 GB). This could be temporary, as you may move this onto Google, but you'll need the raw data and everything extracted before you can send it to the cloud.
4.  create a new branch.
5.  write dev set up guide for the above. Make a documentation folder, name the troubleshooting guide. Basically, make it easy for other people to know how to download the images and get set up.
6.  make a pull request. and assign me to review your changes.
7.  at this point you will then start taking steps to connect this with your google drive so you can work from there and onto collab. Or you will be installing the packages necessary to run the set up on your laptop. Dont start either yet. Tell me which one you are doing and I will give you more instructions from there.

Documentation:

Pre requestee's

This guide used a windows 11 operating system, however may or may not work on other operating systems.

Set up a github account

Install git locally following the instructions from this link: <https://github.com/git-guides/install-git>

1. Restart the computer after installing git (possibly not necessary, this is what I did however)

2. Use The Git Bash terminal for future commands in installation below unless otherwise specified

Installation:

1. Clone everything in the github to the laptop

a. <https://github.com/blobgoat/non-invasive-to-invasive-imagery>

b. using Git Bash run     cd ~/Documents/GitHub_Projects

 i. (replace with your desired path to set the repository in)

c. Run: git clone https://github.com/blobgoat/non-invasive-to-invasive-imagery.git

 i. *Replace with the appropriate url if url has changed

2. When done, go to: <https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images>, to install the image installer. For windows, follow the instructions of the launcher. For other systems, follow the website instructions for installation.

a. Note for Linux commands, Git Bash may not be the appropriate choice for running terminal commands for installation. Choose terminal app appropriately in this case

b. For non-Windows users, the instructions may differ a little bit. Troubleshoot using the website instructions as necessary.

Extract all images

3. Using file explorer, go back to the folder where github folder was downloaded to

a. Go into the file and launch the NBIA data retriever manifest file

b. Accept the terms and conditions

c. Create/browse to a folder you would like the extracted images in.

 i. DO NOT select the folder of the github on your computer

 ii. Browse to the chosen folder in the selection panel

d. Complete the download, which may take several hours. It is recommended to go to computer settings to prevent sleeping, as this can cause errors. (Although, you will also be given an opportunity at the end retry the download for sets with errors. You do not need to restart if you get an error).

 i. Upon prompting at the end, if prompted to retry downloading sets that gave errors click yes.

 ii. **Please note during installation that one set of images early on reported 'Not Authorized'. If you have an authorized account, you may be able to get access to this set. For our purposes, we did not

e. The images should now be extracted into your folder on the computer.

Setting up the git branch

4. Go to the main git page <https://github.com/blobgoat/non-invasive-to-invasive-imagery> (link may change with time)

5. Create a new branch from the "main" repository as the source and give it a name

a. There are plenty of resources on the web that can guide you through this short step.

6. upload documentation to the git branch.   .md files work best, and can be converted from word documents easily with online web tools. Avoid fancy Word features to ensure compatibility when writing documentation.
