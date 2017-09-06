echo -n "Enter Username[140050019]: "
read USERNAME
if [[ -z "$USERNAME" ]]; then
   USERNAME="140050004"
fi

echo -n "Enter Password: "
read -s PASSWD  # -s is there so the characters you type don't show up
curl -i -s --data "uname=${USERNAME}&passwd=${PASSWD}&button=Login" https://internet.iitb.ac.in/index.php --user-agent Mozilla/5.0 > /dev/null