# Section 1:
## Basic operations
equ: ==
not equ: /=

## Functions
:l XXX this loads function XXX
**let** defines a function on the fly

## Lists:
##### 1. some functions/operations:

1. **head/ tail**: head is the first element, tail = all rest

2. **last/ init**: last is the last element, init = all reset

3. **take**:  eg1: take 3 [5,5,4,3,32,3], this takes the first three elements in a list
      eg2: take 10 (cycle [1,2,3])

4. drop: pops out the n elements in a list
elem: eg: 4 `elem` [3,4,5,6] -> this returns True
          elem 4 [3,4,5,6]

5. 5:[]  -> [5]
  1:5:[] -> [1,5]
6. [1 5] !! 2 , this gives 5, the 2nd element in the list

7. long lists: [1..100], [2,4..100] , [1..] can go to infinitely long!
              take 5 [1..] gives [1,2,3,4,5]!!!!

##### 2. list comprehension
  haskell allows a 'set' like representation:
  [x*2 | x <- [1..10]]
  works in a function as well:
  func st=[c | c <- st, c `elem` ['A'...'Z']]
  sometimes, even in nested list:
  [[x | x <- xs, even x] | xs <- xss]

and Lists have to be homogeneous

Tuples:
1. some housekeeping functions
  fst, snd: takes first and second elements in a tuple
  zip: take two lists, and create a list of tuples
2. Tuples can contain elements of different types

Section 2.1: Types
:t returns the type
Types are written in capitals
Int, Float, Double, Bool, Char

Key point:

Types are sets of values, Typecalsses are sets of types.
**Typeclass -> Types -> Values**

Knowledge Point one :
  Functions can take any types -> they can be typeless
Knowledge Point two (Typeclass):
for the (==) operator:
  (==) :: (Eq a) => a -> a-> Bool
This is a 'Equation' class, '=>' is the class constraint, that means 'a' must belongs to the Eq class

Lets walk through all types:
Eq, Ord, Show, Read (read "3" :: Int)
Enum, Bounded, Num, Integral, Floating
## Types
##### Eq
Types that support equality testing
##### Ord
Types that have an ordering
##### Show
class that can be presented as strings
##### Read
members could be Read
##### Enum
types in this class : (), Bool, Char, Ordering,
Int, Integer, Float, Double
##### Bounded
Members that have an upper and lower bound
**minBound** and **maxBound** returns boundary values
##### Num
Numerical typeclass
includes all numbers, real numbers and integral numbers
##### Integral
includes all int values
##### Floating
only floating point, Floats and Doubles

## Syntax in functions

##### pattern matching
  It has to include all possibilities

* eg1:  
  ```python
  lucky :: (Integral a) => a -> String  
  lucky 7 = "LUCKY NUMBER SEVEN!"  
  lucky x = "Sorry, you're out of luck, pal!"
  ```
* eg2:  
```python
addVectors :: (Num a) => (a, a) -> (a, a) -> (a, a)  
addVectors (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)  
```
* eg3:  
```
first :: (a, b, c) -> a  
first (x, _, _) = x
```  
1. _ means 'dont care'
2. type constraint is not required

* eg4:
```
sum' :: (Num a) => [a] -> a  
sum' [] = 0  
sum' (x:xs) = x + sum' xs
```

##### As patterns
```
xs@(x:y:ys)
```
xs is the same as x:y:ys

* eg1:  
```
capital :: String -> String  
capital "" = "Empty string, whoops!"  
capital all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x]
```

##### Guards
```python
bmiTell :: (RealFloat a) => a -> a -> String  
bmiTell weight height  
    | weight / height ^ 2 <= 18.5 = "You're underweight, you emo, you!"  
    | weight / height ^ 2 <= 25.0 = "You're supposedly normal. Pffft, I bet you're ugly!"  
    | weight / height ^ 2 <= 30.0 = "You're fat! Lose some weight, fatty!"  
    | otherwise                 = "You're a whale, congratulations!"  
```
We can use binding keyword **where**  
this **where** binding can be visited by all guards.
it is a type of *global* binding.
```python
bmiTell :: (RealFloat a) => a -> a -> String  
bmiTell weight height  
    | bmi <= skinny = "You're underweight, you emo, you!"  
    | bmi <= normal = "You're supposedly normal. Pffft, I bet you're ugly!"  
    | bmi <= fat    = "You're fat! Lose some weight, fatty!"  
    | otherwise     = "You're a whale, congratulations!"  
    where bmi = weight / height ^ 2  
    (skinny, normal, fat) = (18.5, 25.0, 30.0)  
```

This applies to complex example:
```python
calcBmis :: (RealFloat a) => [(a, a)] -> [a]  
calcBmis xs = [bmi w h | (w, h) <- xs]  
    where bmi weight height = weight / height ^ 2
```
##### Let binding

Form:
```python
let <bindings> in <exp>
```
eg1: basic usage

```python
cylinder :: (RealFloat a) => a -> a -> a  
cylinder r h =
    let sideArea = 2 * pi * r * h  
        topArea = pi * r ^2  
    in  sideArea + 2 * topArea  
```

eg2: introduce function in local scope

```python
 [let square x = x * x in (square 5, square 3, square 2)]
 ```
 ##### Case Statement

 **Syntax form**
 ```python
 case expression of pattern -> result  
                    pattern -> result  
                    pattern -> result  
                    ...  
 ```

 ```python
head' :: [a] -> a  
head' xs = case xs of [] -> error "No head for empty lists!"  
                      (x:_) -> x
 ```

##### Try these templates:
1.
```
pattern
  | expression = result
  ...
  | otherwise = result
```
2.
```
result where
pattern = result
...
```
3.
```
let pattern = result
...
in result
```
4.
```
case expression of pattern -> result
      ...
```
##### Higher Order functions
The functions in Haskell can be **partially applied**

Consider the two cases:
case 1:
```
compareWithHundred :: (Num a, Ord a) => a -> Ordering  
compareWithHundred x = compare 100 x
```
case 2:
```
compareWithHundred :: (Num a, Ord a) => a -> Ordering  
compareWithHundred = compare 100  
```
Note:
There's a special case  - (-4);
this is used to represent numerical value of negative 4.
To represent subtract 4, we use (**subtract 4**)
More examples:
eg1:
```
isUpperAlphanum :: Char -> Bool  
isUpperAlphanum = (`elem` ['A'..'Z'])
```

### High Orderism###
 **Eg1**
```
applyTwice :: (a -> a) -> a -> a  
applyTwice f x = f (f x)  
```
1. (a->a) corresponds to a function that takes 'a' as input and also returns type 'a'.
2. -> sign is right associative

**Eg2**
```
zipWith' :: (a -> b -> c) -> [a] -> [b] -> [c]  
zipWith' _ [] _ = []  
zipWith' _ _ [] = []  
zipWith' f (x:xs) (y:ys) = f x y : zipWith' f xs ys  
```
1. In this case, (a -> b -> c) represents a function.
2. (x:xs) represents a list with head x and rest xs

**Eg3**
```
flip' :: (a -> b -> c) -> (b -> a -> c)  
flip' f = g  
    where g x y = f y x  
```
we can improve this
```
flip' :: (a -> b -> c) -> b -> a -> c  
flip' f y x = f x y
```

### Map and Filters###
**Map**

Map function applies a function to every element in a list:
```
map :: (a -> b) -> [a] -> [b]  
map _ [] = []  
map f (x:xs) = f x : map f xs  
```

**Filter**
```
filter :: (a -> Bool) -> [a] -> [a]  
filter _ [] = []  
filter p (x:xs)   
    | p x       = x : filter p xs  
    | otherwise = filter p xs
```

Example:
```
largestDivisible :: (Integral a) => a  
largestDivisible = head (filter p [100000,99999..])  
    where p x = x `mod` 3829 == 0  
```
In this case, p is the conditional function for filter.

**takeWhile**
takeWhile (condition) [list]
this functions returns a list that all elements within the list satisfies the given condition
eg:
```
sum (takeWhile (<10000) (filter odd (map (^2) [1..])))
```

### Lambdas
it's a shorthand for defining functions
eg1:
```
zipWith (\a b -> (a * 30 + 3) / b) [5,4,3,2,1] [1,2,3,4,5]
```
eg2:
```
flip' :: (a -> b -> c) -> b -> a -> c  
flip' f = \x y -> f y x  
```

### Folds and horses

###### foldl
eg 1
```
sum' :: (Num a) => [a] -> a  
sum' xs = foldl (\acc x -> acc + x) 0 xs  
```
1. acc is initialized to be 0
2. x keeps taking value from list xs

eg 2
```
elem' :: (Eq a) => a -> [a] -> Bool  
elem' y ys = foldl (\acc x -> if x == y then True else acc) False ys
```

eg 3
```
map' :: (a -> b) -> [a] -> [b]  
map' f xs = foldr (\x acc -> f x : acc) [] xs
```
1. This is fold right, starting from the right of a list
2. acc takes [] and x take xs

**Folds can be used to implement any function where you traverse a list once, element by element, and then return something based on that. Whenever you want to traverse a list to return something, chances are you want a fold.**

Note:
1. **foldl1** and **foldr1**: they work exactly the same as foldl and foldr, but they exe
eg:
```
maximum' :: (Ord a) => [a] -> a  
maximum' = foldr1 (\x acc -> if x > acc then x else acc)  

reverse' :: [a] -> [a]  
reverse' = foldl (\acc x -> x : acc) []
```
2. **scanl** and **scanr**, they show the intermediate values
```
ghci> scanl (+) 0 [3,5,2,1]  
[0,3,8,10,11]  
ghci> scanr (+) 0 [3,5,2,1]  
[11,8,3,1,0]  
ghci> scanl1 (\acc x -> if x > acc then x else acc) [3,4,5,3,7,9,2,1]  
[3,4,5,5,7,9,9,9]  
ghci> scanl (flip (:)) [] [3,2,1]  
[[],[3],[2,3],[1,2,3]]
```

### Dollar sign
1. Lowest precedence
2. right-associative, f (g (z x)) is f $ g $ z x

```
sqrt (3 + 4 + 9)
```
is equivalent to
```
sqrt $ 3+4+9
```
### Function composition
This is represented by the (.) symbol

```
(.) :: (b -> c) -> (a -> b) -> a -> c  
f . g = \x -> f (g x)
```

Note:
1. The input of f must have the same type as input of g
2. Combine functions
eg :
It used to be:
```
map (\x -> negate (abs x)) [5,-3,-6,7,-3,2,-19,24]
```
Now it is:
```
map (negate . abs) [5,-3,-6,7,-3,2,-19,24]
```
It even works for functions take multiple inputs:
It used to be:
```
sum (replicate 5 (max 6.7 8.9))
```
Now it is:
```
(sum . replicate 5 . max 6.7) 8.9
```
or
```
sum . replicate 5 . max 6.7 $ 8.9
```

The followings are equivalent
1.
```
oddSquareSum :: Integer  
oddSquareSum = sum (takeWhile (<10000) (filter odd (map (^2) [1..])))
```
2.
```
oddSquareSum :: Integer  
oddSquareSum = sum . takeWhile (<10000) . filter odd . map (^2) $ [1..]
```
3.
```
oddSquareSum :: Integer  
oddSquareSum =   
    let oddSquares = filter odd $ map (^2) [1..]  
        belowLimit = takeWhile (<10000) oddSquares  
    in  sum belowLimit  
```
The last one is easier to read with a let binding
