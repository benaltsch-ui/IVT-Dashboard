# IVT Dashboard

## Resolving GitHub merge conflicts
If GitHub shows a "This branch has conflicts" warning when opening a pull request, it means the base branch changed after your branch was created and both versions edit the same part of a file (commonly `ai_studio_code.py`). To resolve:

1. Add and fetch the remote that hosts the base branch, for example:
   ```bash
   git remote add origin <your-remote-url>
   git fetch origin
   ```
2. Update your feature branch with the latest base branch commits (replace `main` if the base branch is named differently):
   ```bash
   git checkout work
   git merge origin/main
   ```
   If you prefer a linear history, use `git rebase origin/main` instead of merge.
3. Open any files marked as conflicted and keep the correct version of the code. Remove the `<<<<<<<`, `=======`, and `>>>>>>>` markers, then save the cleaned file.
4. Run your checks (for example `python -m compileall ai_studio_code.py`) to confirm the code still works.
5. Commit the resolved changes and push the branch:
   ```bash
   git add <files>
   git commit -m "Resolve merge conflicts"
   git push origin work
   ```

After the push, refresh the pull request; the conflicts banner should disappear.
