import base64
from github import Github
from github import InputGitTreeElement
import pdb

def push_git(load_n_test,csv_,src_path):
    # pdb.set_trace()
    access_token="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"
    g = Github(access_token)
    repo = g.get_user().get_repo('git-test') # repo name

    commit_message = 'python commit'
    master_ref = repo.get_git_ref('heads/master')
    master_sha = master_ref.object.sha
    base_tree = repo.get_git_tree(master_sha)

    element_list = list()
    entry = 'c.txt'
    with open(entry, "a+") as f:
        str_ = load_n_test+' , '+csv_+' , '+src_path
        f.write(str_+'\n')
        f.seek(0)
        data = f.read()
    file_name = entry.split('/')[-1]
    element = InputGitTreeElement(file_name, '100644', 'blob', data)
    element_list.append(element)

    tree = repo.create_git_tree(element_list, base_tree)
    parent = repo.get_git_commit(master_sha)
    commit = repo.create_git_commit(commit_message, tree, [parent])
    master_ref.edit(commit.sha)

# pdb.set_trace()
if __name__ == "__main__":
    for i in range(10):
        if(i%2==0):
            push_git('e.pth','e.csv','e')
        else:
            push_git('f.pth','f.csv','f')

