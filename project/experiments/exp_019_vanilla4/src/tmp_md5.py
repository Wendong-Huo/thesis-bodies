  
import hashlib 
  
# encoding GeeksforGeeks using md5 hash 
# function  
  
# initializing string 
str2hash = "GeeksforGeeks"
  
# encoding GeeksforGeeks using encode() 
# then sending to md5() 
result = hashlib.md5(str2hash.encode())
  
# printing the equivalent hexadecimal value. 
print("The hexadecimal equivalent of hash is : ", end ="") 
print(result.hexdigest()) 