{-# OPTIONS_GHC -Wno-unused-top-binds #-}
module Main (main) where

-- 4
-- palindromo :: (Eq a) => [a] -> Bool
-- palindromo [] = True
-- palindromo xs = xs == reverse xs

-- 5
-- mult :: Int -> Int -> Int -> Int
-- mult = \x -> \y -> \z -> x * y * z

-- 6
-- 1
-- False || False = False
-- False || True = True
-- True || False = True
-- True || True = True
-- -- 2
-- False || False = False
-- _ || _ = True
-- -- 3
-- False || b = b
-- True || _ = True
-- -- 4
-- b || c | b == c = b
-- | otherwise = True

-- 7
e' x y = if x then (if y then True else False) else False

-- 8
ee x y = if x then y else False

-- 9
c :: a -> b -> a
c x y = x

-- 10
co :: (b -> c) -> (a -> b) -> a -> c
co f g x = f (g x)

-- 11
penultimo :: [a] -> a
penultimo [] = error "Lista vazia"
penultimo [x] = error "Lista com um elemento"
penultimo xs = reverse xs !! 1

--12
maximoLocal :: [Int] -> [Int]
maximoLocal (x:y:z:xs) = if (y > x && y > z) then (y : maximoLocal (z:xs)) else (maximoLocal (y:z:xs))
maximoLocal _ = []

-- 13
fatores :: Int -> [Int]
fatores n = [x | x <- [1..n-1], n `mod` x == 0]

perfeitos :: Int -> [Int]
perfeitos n = [x | x <- [1..n], (sum (fatores x)) == x]

-- 14
produtoEscalar :: Num a => [a] -> [a] -> a
produtoEscalar xs ys = sum [x * y | (x, y) <- zip xs ys]

-- 15
palindromo :: [Int] -> Bool
palindromo [] = True
palindromo [_] = True
palindromo xs = (head xs == last xs) && palindromo (tail (init xs))

-- 16
ordenaListas :: (Num a, Ord a) => [[a]] -> [[a]]
ordenaListas [] = []
ordenaListas (x:xs) = ordenaListas menores ++ [x] ++ ordenaListas maiores
  where
    menores = [y | y <- xs, length y < length x]
    maiores = [y | y <- xs, length y > length x]

-- 17
coord :: [a] -> [a] -> [(a,a)]
coord x y = concat [[(i, j) | i <- x] | j <- y]

-- 18
digitosRev :: Int -> [Int]
digitosRev 0 = []
digitosRev x = x `mod` 10 : digitosRev ( x `div` 10)

dobroAlternado :: [Int] -> [Int]
dobroAlternado [] = []
dobroAlternado [x] = [x]
dobroAlternado (x:y:xs) = [x] ++ [2*y] ++ dobroAlternado xs

somaDigitos :: [Int] -> Int
somaDigitos [] = 0
somaDigitos [x] = x
somaDigitos (x:y:xs) = if x > 9 || y > 9 then somaDigitos (x `mod` 10 + x `div` 10 : y`mod`10 + y `div` 10 : xs) else  x + y + somaDigitos xs

luhn :: Int -> Bool
luhn x = result `mod` 10 == 0
  where result = (somaDigitos . dobroAlternado.  digitosRev) x

-- 19
mc91 :: Integral a => a -> a
mc91 n | n > 100 = n - 10
       | otherwise = (mc91 . mc91) (n + 11)

-- 20
elem' :: Eq a => a -> [a] -> Bool
elem' _ [] = False
elem' n (x:xs) = n == x || elem' n xs

-- 21
euclid :: Int -> Int -> Int
euclid x y | x == y = x
           | x > y = euclid (x-y) y
           | otherwise = euclid x (y-x)

-- 22
concat' :: [[a]] -> [a]
concat' [] = []
concat' (xs:xss) = [x | x <- xs] ++ concat' xss

-- 23
intersperse' :: a -> [a] -> [a]
intersperse' _ [] = []
intersperse' _ [x] = [x]
intersperse' separator (char:str) = char : separator : intersperse' separator str

-- 24
digitos :: Int -> [Int]
digitos 0 = []
digitos x = digitos ( x `div` 10) ++ [x `mod` 10]

digitsName :: [Int] -> [String]
digitsName [] = []
digitsName (x:xs) | x == 1 = "um" : digitsName xs
                  | x == 2 = "dois" : digitsName xs
                  | x == 3 = "tres" : digitsName xs
                  | x == 4 = "quatro" : digitsName xs
                  | x == 5 = "cinco" : digitsName xs
                  | x == 6 = "seis" : digitsName xs
                  | x == 7 = "sete" : digitsName xs
                  | x == 8 = "oito" : digitsName xs
                  | x == 9 = "nove" : digitsName xs
                  | x == 0 = "zero" : digitsName xs
                  | otherwise = []

wordNumber :: Char -> Int -> [Char]
wordNumber separator n = concat' $ intersperse' [separator] (digitsName $ digitos n)

-- Quiz manhã
zipe :: ([a], [b]) -> [(a, b)]
zipe ([], _) = []
zipe (_, []) = []
zipe (x:xs, y:ys) = (x, y) : zipe (xs, ys)

zipWithe :: (a -> b -> c) -> ([a], [b]) -> [c]
zipWithe _ ([], _) = []
zipWithe _ (_, []) = []
zipWithe f (x:xs, y:ys) = f x y : zipWithe f (xs, ys)

main :: IO ()
main = do
  let x = zipWithe (\x -> \y -> (x, y, x+y)) ([1, 2, 3], [4, 5, 6])

  putStrLn (show x)
