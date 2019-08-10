# Materials

* [How To Install Apache Tomcat 8 on Ubuntu 16.04 @ digitalocean](https://www.digitalocean.com/community/tutorials/how-to-install-apache-tomcat-8-on-ubuntu-16-04)

# Install on Ubuntu

```bash
# Install Java
> sudo apt-get update
> sudo apt-get install default-jdk

# Create Tomcat User
> sudo groupadd tomcat
> sudo useradd -s /bin/false -g tomcat -d /opt/tomcat tomcat

# Install Tomcat
> cd /tmp
> curl -O http://apache.mirrors.ionfish.org/tomcat/tomcat-8/v8.5.5/bin/apache-tomcat-8.5.5.tar.gz
> sudo mkdir /opt/tomcat
> sudo tar xzvf apache-tomcat-8*tar.gz -C /opt/tomcat --strip-components=1

# Update Permissions
> cd /opt/tomcat
> sudo chgrp -R tomcat /opt/tomcat
> sudo chmod -R g+r conf
> sudo chmod g+x conf
> sudo chown -R tomcat webapps/ work/ temp/ logs/

# Create a systemd Service File
> sudo update-java-alternatives -l
> sudo nano /etc/systemd/system/tomcat.service

[Unit]
Description=Apache Tomcat Web Application Container
After=network.target

[Service]
Type=forking

Environment=JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre
Environment=CATALINA_PID=/opt/tomcat/temp/tomcat.pid
Environment=CATALINA_HOME=/opt/tomcat
Environment=CATALINA_BASE=/opt/tomcat
Environment='CATALINA_OPTS=-Xms512M -Xmx1024M -server -XX:+UseParallelGC'
Environment='JAVA_OPTS=-Djava.awt.headless=true -Djava.security.egd=file:/dev/./urandom'

ExecStart=/opt/tomcat/bin/startup.sh
ExecStop=/opt/tomcat/bin/shutdown.sh

User=tomcat
Group=tomcat
UMask=0007
RestartSec=10
Restart=always

[Install]
WantedBy=multi-user.target

> sudo systemctl daemon-reload
> sudo systemctl start tomcat
> sudo systemctl status tomcat

#  Adjust the Firewall and Test the Tomcat Server
> sudo ufw allow 8080
> sudo systemctl enable tomcat

# Configure Tomcat Web Management Interface
> sudo nano /opt/tomcat/conf/tomcat-users.xml

<tomcat-users . . .>
    <user username="admin" password="password" roles="manager-gui,admin-gui"/>
</tomcat-users>

> sudo nano /opt/tomcat/webapps/manager/META-INF/context.xml
> sudo nano /opt/tomcat/webapps/host-manager/META-INF/context.xml

<Context antiResourceLocking="false" privileged="true" >
  <!--<Valve className="org.apache.catalina.valves.RemoteAddrValve"
         allow="127\.\d+\.\d+\.\d+|::1|0:0:0:0:0:0:0:1" />-->
</Context>

> sudo systemctl restart tomcat

# Access the Web Interface
# http://server_domain_or_IP:8080
# http://server_domain_or_IP:8080/manager/html
# http://server_domain_or_IP:8080/host-manager/html/
```
