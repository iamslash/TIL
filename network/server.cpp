/* Copyright (C) 2020 by iamslash */

#include<stdio.h> 
#include<sys/types.h> 
#include<sys/socket.h> 
#include<sys/un.h> 
#include<string.h> 
#include<netdb.h> 
#include<netinet/in.h> 
#include<arpa/inet.h> 
#include<string.h> 

#define SERVER_ADDR "10.1.2.3"
int main() 
{ 
  char sendBuf[256], recvBuf[256]; 
  int fd = socket(AF_INET, SOCK_STREAM, 0); 
  if (fd < 0) 
    printf("Error in server creating\n"); 
  else
    printf("Server Created\n"); 
          
  struct sockaddr_in serverAddr, peerAddr; 
  serverAddr.sin_family = AF_INET; 
  serverAddr.sin_addr.s_addr = INADDR_ANY; 
  serverAddr.sin_addr.s_addr = inet_addr(SERVER_ADDR); 
  serverAddr.sin_port = htons(12000); 
  
  if (bind(fd, (struct sockaddr*) &serverAddr, sizeof(serverAddr)) == 0) 
    printf("Binded Correctly\n"); 
  else
    printf("Unable to bind\n"); 
          
  if (listen(fd, 3) == 0) 
    printf("Listening ...\n"); 
  else
    printf("Unable to listen\n"); 
      
  socklen_t addr_size; 
  addr_size = sizeof(struct sockaddr_in); 

  char *ip; 
      
  while (1) 
  { 
    int acc = accept(fd, (struct sockaddr*) &peerAddr, &addr_size); 
    printf("Connection Established\n"); 
    char ip[INET_ADDRSTRLEN]; 
    inet_ntop(AF_INET, &(peerAddr.sin_addr), ip, INET_ADDRSTRLEN); 

    printf("connection established with IP : %s and PORT : %d\n",  
           ip, ntohs(peerAddr.sin_port)); 
  
    recv(acc, recvBuf, 256, 0); 
    printf("Client : %s\n", recvBuf); 
    strcpy(sendBuf, "Hello"); 
    send(acc, sendBuf, 256, 0); 
  }  
  return 0; 
} 
