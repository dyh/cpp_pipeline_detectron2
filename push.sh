echo "#################### config global user name & email ####################"
git config --global user.email "1358366+dyh@users.noreply.github.com"
git config --global user.name "dyh"

echo "#################### git add . ####################"
git add .

echo "#################### git pull ####################"
git pull

echo "#################### git commit -m \"daily\" ####################"
git commit -m "daily"

echo "#################### git push -u origin master ####################"
git push -u origin main

echo "#################### done ####################"

