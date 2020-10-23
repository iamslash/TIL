/* Copyright (C) 2020 by iamslash */

#include<stdio.h> 
#include<sys/types.h> 
#include<sys/socket.h> 
#include<sys/un.h> 
#include<string.h> 
#include<netdb.h> 
#include<netinet/in.h> 
#include<arpa/inet.h> 
#include<stdlib.h> 

#define SERVER_ADDR "10.1.2.3"
#define CLIENT_ADDR "10.1.2.2"

int main() 
{ 
  char recvBuf[256], sendBuf[256]; 
  struct sockaddr_in serverAddr, clientAddr; 
  int fd = socket(AF_INET, SOCK_STREAM, 0); 
  if (fd < 0) 
    printf("Error in client creating\n"); 
  else
    printf("Client Created\n"); 
          
  serverAddr.sin_family = AF_INET; 
  serverAddr.sin_addr.s_addr = INADDR_ANY; 
  serverAddr.sin_port = htons(12000); 
  serverAddr.sin_addr.s_addr = inet_addr(SERVER_ADDR); 
  
  clientAddr.sin_family = AF_INET; 
  clientAddr.sin_addr.s_addr = INADDR_ANY; 
  clientAddr.sin_port = htons(12010); 

  clientAddr.sin_addr.s_addr = inet_addr(CLIENT_ADDR); 
  if (bind(fd, (struct sockaddr*) &clientAddr, sizeof(struct sockaddr_in)) == 0) 
    printf("Binded Correctly\n"); 
  else
    printf("Unable to bind\n");

  socklen_t addr_size = sizeof serverAddr;
  int con = connect(fd, (struct sockaddr*) &serverAddr, sizeof serverAddr); 
  if (con == 0) 
    printf("Client Connected\n"); 
  else
    printf("Error in Connection\n"); 
  
  strcpy(sendBuf, "Hello World"); 
  send(fd, sendBuf, 256, 0);  
  recv(fd, recvBuf, 256, 0); 
  printf("Received : %s\n", recvBuf); 
  return 0; 
} 
