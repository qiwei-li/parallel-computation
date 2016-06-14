parquad<- function(u,a,val){
  library(Rdsm)
  myidxs = getidxs(length(u)) 
  temp<-t(u[,])[,myidxs] %*% a[myidxs,]%*%u[,]
  rdsmlock("vallock")
  val[,] = val[,] + temp
  rdsmunlock("vallock")
  0
}

test<-function(cls){
  library(parallel)
  library(Rdsm)
  mgrinit(cls)
  mgrmakevar(cls,"u",8,1) 
  mgrmakevar(cls,"a",8,8) 
  mgrmakevar(cls,"val",1,1) 
  mgrmakevar(cls,"vallock",1,1)
  
  uu = 11:18
  aa = matrix(0, nrow=8, ncol=8)
  for(i in 1:8){
    for(j in i:8){
      aa[i,j] = j
    }
  }
  aa = aa + t(aa)
  correct =  t(as.matrix(uu)) %*% as.matrix(aa) %*% as.matrix(uu)
  print(correct)
  
  u[,]<- 11:18
  a[,]<- as.numeric(aa)
  val[,]<-0
  print(u[,])
  print(a[,])
  mgrmakelock(cls,"vallock")
  clusterExport(cls,"parquad")
  clusterEvalQ(cls, parquad(u,a,val)) 
  print(val[,])
}


library(parallel)
cls = makeCluster(5)
test(cls)
stopCluster(cls)





# uuu = as.matrix(c(1:4))
# aaa = matrix(1:16, nrow=4, ncol=4)
# correct = t(uuu)%*%aaa%*%uuu
# correct



  
