Author/User: the21catt

Date Finished: 8/1/2025

## Goals

1.  Have git set up and have everything cloned to your laptop.

3.  Extract all images. It requires 100 GB. Note: It is possible to set up on Google collab but you'll at least need 100 GB (the raw data) and everything extracted before you can send it to the cloud.
4.  create a new branch.
5.  revise dev set up guide for the above with any difficulties you encountered. And fix typos or suggest formatting. Basically, make it easy for other people to know how to download the images and get set up.
6.  make a pull request. and assign a senior collaborator (like the esteemed blobgoat) to review your changes. If no changes should be made, still create a pull request (to replace last modified data)

## Documentation:

For geting set up on windows, read, *windows_setup*.
**Note** this will take several hours, so plan accordingly.

### Extract all images

1. change settings on your pc to disable sleeping (this will take several hours and you likely will want to leave your pc or laptop)

2.  install <https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images>. There is a .tcia file inside the data folder- so **no need to find the exact images**. You just need the installer to install all the addresses within the file.

3. Using file explorer, go back to the folder where github folder was downloaded to. 
    
    **EXPECT THIS TO TAKE SEVERAL HOURS**

    * Go into the file and launch the NBIA data retriever manifest file
    * Accept the terms and conditions
    * browse to your git folder which should be named non-invasive-imagery-to-invasive-imagery.
    * navigate to ./non-invasive-imagery-to-invasive-imagery/data

    * Name it **Images** and place it in your data folder within your document
* **Browse to the chosen folder in the** ***selection panel***

Complete the download, which may take several hours. If you forget to turn off sleeping, going to sleep will throw errors. Don't fear if you forgot, you will also be given an opportunity at the end retry the download for sets with errors. **You do not need to restart if you get an error**.

* Upon prompting at the end, if prompted to retry downloading sets that gave errors, click **yes**.

- **Please note** during installation that one set of images early on reported 'Not Authorized'. If you have an authorized account, you may be able to get access to this set. For our purposes, we did not.

Once done, the images should now be extracted into your folder on the computer.

## Setting up the git branch
Skip this step if you are familiar with setting up a git branch

4. Go to the main git page <https://github.com/blobgoat/non-invasive-to-invasive-imagery> (link may change with time)

5. Create a new branch from the "main" repository as the source and give it a name
    -There are plenty of resources on the web that can guide you through this short step.

6. Format and upload documentation to the git branch.   .md files work best, and can be converted from word documents easily with online web tools. Avoid fancy Word features to ensure compatibility when writing documentation.
Guides on how to format bolding, italics, bullet points, and headings can be found online
Here is github's page on it: <https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax>
