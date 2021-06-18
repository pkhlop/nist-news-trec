sudo ls;
./gradlew :distribution:packages:deb:assemble;
sudo service elasticsearch stop;
sudo dpkg -i ./distribution/packages/deb/build/distributions/elasticsearch-7.8.2-SNAPSHOT-amd64.deb;
echo "elasticsearch hold"| sudo dpkg --set-selections
sudo systemctl enable elasticsearch.service
sudo service elasticsearch start;
sudo service elasticsearch status;
