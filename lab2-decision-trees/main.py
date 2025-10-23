from . import id3, c45

if __name__ == "__main__":
  print("[ID3]".center(20, "="))
  id3.run_algo()
  
  print("[C4.5]".center(20, "="))
  c45.run_algo()
