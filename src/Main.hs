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
-- e' x y = if x then (if y then True else False) else False

-- 8
-- ee x y = if x then y else False

-- 9
-- c :: a -> b -> a
-- c x y = x

-- 10
-- co :: (b -> c) -> (a -> b) -> a -> c
-- co f g x = f (g x)

-- 11
-- penultimo :: [a] -> a
-- penultimo [] = error "Lista vazia"
-- penultimo [x] = error "Lista com um elemento"
-- penultimo xs = reverse xs !! 1

--12
-- maximoLocal :: [Int] -> [Int]
-- maximoLocal (x:y:z:xs) = if (y > x && y > z) then (y : maximoLocal (z:xs)) else (maximoLocal (y:z:xs))
-- maximoLocal _ = []

-- 13
-- fatores :: Int -> [Int]
-- fatores n = [x | x <- [1..n-1], n `mod` x == 0]

-- perfeitos :: Int -> [Int]
-- perfeitos n = [x | x <- [1..n], (sum (fatores x)) == x]

-- 14
-- produtoEscalar :: Num a => [a] -> [a] -> a
-- produtoEscalar xs ys = sum [x * y | (x, y) <- zip xs ys]

-- 15
-- palindromo :: [Int] -> Bool
-- palindromo [] = True
-- palindromo [_] = True
-- palindromo xs = (head xs == last xs) && palindromo (tail (init xs))

-- 16
-- ordenaListas :: (Num a, Ord a) => [[a]] -> [[a]]
-- ordenaListas [] = []
-- ordenaListas (x:xs) = ordenaListas menores ++ [x] ++ ordenaListas maiores
--   where
--     menores = [y | y <- xs, length y < length x]
--     maiores = [y | y <- xs, length y > length x]

-- ordenaListas :: (Num a, Ord a) => [[a]] -> [[a]]
-- ordenaListas [] = []
-- ordenaListas [x] = [x]
-- ordenaListas (x1:x2:xs) = if length x1 > length x2 then [x2] ++ [x1] ++ ordenaListas xs else ordenaListas (x1:x2:xs)

-- 17
coord :: [a] -> [a] -> [(a,a)]
coord x y = [(i,j) | i <- x, j <- y]

main :: IO ()
main = do
  let x = ordenaListas [[4, 3, 2], [2, 2, 1, 3, 4], [1], [2, 3, 3, 2, 2]]
  putStrLn (show x)
