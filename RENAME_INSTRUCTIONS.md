# Repository Rename Instructions

## Current Repository Name
`Face-embedding-generator`

## New Repository Name
`yunet_alignment_face_model`

## Steps to Rename on GitHub:

1. Go to: https://github.com/ay-ush-17/Face-embedding-generator
2. Click on **Settings** tab
3. In the **Repository name** field, change to: `yunet_alignment_face_model`
4. Click **Rename** button
5. GitHub will automatically redirect and update

## After Renaming on GitHub, Update Local Repository:

Run these commands in your terminal:

```powershell
cd "d:\MY WORK\EZ pic"

# Update the remote URL to the new repository name
git remote set-url origin https://github.com/ay-ush-17/yunet_alignment_face_model.git

# Verify the change
git remote -v

# Test the connection
git fetch origin
```

## Note:
- GitHub automatically redirects from old name to new name
- Existing clones will continue to work
- All issues, PRs, and stars are preserved
- Update any documentation or links that reference the old name

## Repository Description (Suggested):
"Face recognition pipeline using YuNet face detection, MTCNN-style alignment, and MobileFaceNet embeddings with GUI applications for testing and debugging"

## Current Status:
✅ All changes committed to yunet-detection-test branch
✅ Changes pushed to GitHub
⏳ Repository rename pending on GitHub website
