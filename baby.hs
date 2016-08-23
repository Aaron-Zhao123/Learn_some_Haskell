maximum' :: (Ord a) => [a] -> a
maximum' [] = error "maximum of empty list"
maximum' [x] = x
maximum' x:xs
  | x > maxtail = x
  | otherwis = maxtail
  where maxtail = maximum' xs
