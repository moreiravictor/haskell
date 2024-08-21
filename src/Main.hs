{-# OPTIONS_GHC -Wno-unused-top-binds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE InstanceSigs #-}
module Main (main) where
-- -- lista 2

-- -- 4
-- -- palindromo :: (Eq a) => [a] -> Bool
-- -- palindromo [] = True
-- -- palindromo xs = xs == reverse xs

-- -- 5
-- -- mult :: Int -> Int -> Int -> Int
-- -- mult = \x -> \y -> \z -> x * y * z

-- -- 6
-- -- 1
-- -- False || False = False
-- -- False || True = True
-- -- True || False = True
-- -- True || True = True
-- -- -- 2
-- -- False || False = False
-- -- _ || _ = True
-- -- -- 3
-- -- False || b = b
-- -- True || _ = True
-- -- -- 4
-- -- b || c | b == c = b
-- -- | otherwise = True

-- -- 7
-- e' x y = if x && y then True else False

-- -- 8
-- ee x y = x && y

-- -- 9
-- c :: a -> b -> a
-- c x y = x

-- -- 10
-- co :: (b -> c) -> (a -> b) -> a -> c
-- co f g x = f (g x)

-- -- 11
-- penultimo :: [a] -> a
-- penultimo [] = error "Lista vazia"
-- penultimo [x] = error "Lista com um elemento"
-- penultimo xs = reverse xs !! 1

-- --12
-- maximoLocal :: [Int] -> [Int]
-- maximoLocal (x:y:z:xs) = if y > x && y > z then y : maximoLocal (z:xs) else maximoLocal (y:z:xs)
-- maximoLocal _ = []

-- -- 13
-- fatores :: Int -> [Int]
-- fatores n = [x | x <- [1..n-1], n `mod` x == 0]

-- perfeitos :: Int -> [Int]
-- perfeitos n = [x | x <- [1..n], sum (fatores x) == x]

-- -- 14
-- produtoEscalar :: Num a => [a] -> [a] -> a
-- produtoEscalar xs ys = sum [x * y | (x, y) <- zip xs ys]

-- -- 15
-- palindromo :: [Int] -> Bool
-- palindromo [] = True
-- palindromo [_] = True
-- palindromo xs = head xs == last xs && palindromo (tail (init xs))

-- -- 16
-- ordenaListas :: (Num a, Ord a) => [[a]] -> [[a]]
-- ordenaListas [] = []
-- ordenaListas (x:xs) = ordenaListas menores ++ [x] ++ ordenaListas maiores
--   where
--     menores = [y | y <- xs, length y < length x]
--     maiores = [y | y <- xs, length y > length x]

-- -- 17
-- coord :: [a] -> [a] -> [(a,a)]
-- coord x y = concat [[(i, j) | i <- x] | j <- y]

-- -- 18
-- digitosRev :: Int -> [Int]
-- digitosRev 0 = []
-- digitosRev x = x `mod` 10 : digitosRev ( x `div` 10)

-- dobroAlternado :: [Int] -> [Int]
-- dobroAlternado [] = []
-- dobroAlternado [x] = [x]
-- dobroAlternado (x:y:xs) = [x] ++ [2*y] ++ dobroAlternado xs

-- somaDigitos :: [Int] -> Int
-- somaDigitos [] = 0
-- somaDigitos [x] = x
-- somaDigitos (x:y:xs) = if x > 9 || y > 9 then somaDigitos (x `mod` 10 + x `div` 10 : y`mod`10 + y `div` 10 : xs) else  x + y + somaDigitos xs

-- luhn :: Int -> Bool
-- luhn x = result `mod` 10 == 0
--   where result = (somaDigitos . dobroAlternado.  digitosRev) x

-- -- 19
-- mc91 :: Integral a => a -> a
-- mc91 n | n > 100 = n - 10
--        | otherwise = (mc91 . mc91) (n + 11)

-- -- 20
-- elem' :: Eq a => a -> [a] -> Bool
-- elem' _ [] = False
-- elem' n (x:xs) = n == x || elem' n xs

-- -- 21
-- euclid :: Int -> Int -> Int
-- euclid x y | x == y = x
--            | x > y = euclid (x-y) y
--            | otherwise = euclid x (y-x)

-- -- 22
-- concat' :: [[a]] -> [a]
-- concat' xss = foldr (\ xs -> (++) ([x | x <- xs])) [] xss

-- -- 23
-- intersperse' :: a -> [a] -> [a]
-- intersperse' _ [] = []
-- intersperse' _ [x] = [x]
-- intersperse' separator (char:str) = char : separator : intersperse' separator str

-- -- 24
-- digitos :: Int -> [Int]
-- digitos 0 = []
-- digitos x = digitos ( x `div` 10) ++ [x `mod` 10]

-- digitsName :: [Int] -> [String]
-- digitsName [] = []
-- digitsName (x:xs) | x == 1 = "um" : digitsName xs
--                   | x == 2 = "dois" : digitsName xs
--                   | x == 3 = "tres" : digitsName xs
--                   | x == 4 = "quatro" : digitsName xs
--                   | x == 5 = "cinco" : digitsName xs
--                   | x == 6 = "seis" : digitsName xs
--                   | x == 7 = "sete" : digitsName xs
--                   | x == 8 = "oito" : digitsName xs
--                   | x == 9 = "nove" : digitsName xs
--                   | x == 0 = "zero" : digitsName xs
--                   | otherwise = []

-- wordNumber :: Char -> Int -> String
-- wordNumber separator n = concat' $ intersperse' [separator] (digitsName $ digitos n)

-- -- Quiz manhã
-- zipe :: ([a], [b]) -> [(a, b)]
-- zipe ([], _) = []
-- zipe (_, []) = []
-- zipe (x:xs, y:ys) = (x, y) : zipe (xs, ys)

-- zipWithe :: (a -> b -> c) -> ([a], [b]) -> [c]
-- zipWithe _ ([], _) = []
-- zipWithe _ (_, []) = []
-- zipWithe f (x:xs, y:ys) = f x y : zipWithe f (xs, ys)

-- -- lista 3

-- -- 1
-- data Nat = Zero | Succ Nat
--   deriving Show
-- add :: Nat -> Nat -> Nat
-- add Zero n = n
-- add (Succ m) n = Succ (add m n)

-- mult :: Nat -> Nat -> Nat
-- mult Zero _ = Zero
-- mult (Succ m) n = add n (mult m n)

-- -- 2
-- data Tree a = Leaf a | Node (Tree a) a (Tree a)
--   deriving (Show)

-- occurs :: Ord a => a -> Tree a -> Bool
-- occurs n (Leaf m) = n == m
-- occurs n (Node l m r) = case compare n m of
--                     EQ -> True
--                     LT -> occurs n l
--                     GT -> occurs n r

-- -- 3
-- flatten :: Tree a -> [a]
-- flatten (Leaf m) = [m]
-- flatten (Node l m r) = flatten l ++ [m] ++ flatten r


-- -- 4
-- data BTree a = BLeaf a | BNode (BTree a) (BTree a)
--   deriving Show
-- leafNum :: BTree a -> Int
-- leafNum (BLeaf _) = 1
-- leafNum (BNode l r) = leafNum l + leafNum r

-- balanced :: BTree a -> Bool
-- balanced (BLeaf _) = True
-- balanced (BNode l r) = abs (leafNum l - leafNum r) <= 1 && balanced l && balanced r

-- -- 5
-- divide :: [a] -> ([a], [a])
-- divide xs = splitAt middle xs
--         where
--           middle = length xs `div` 2

-- balance :: [a] -> BTree a
-- balance [x] = BLeaf x
-- balance xs = BNode (balance l) (balance r)
--           where
--             (l, r) = divide xs

-- -- 6
-- data Expr = Val Int | Add Expr Expr

-- folde :: (Int -> a) -> (a -> a -> a) -> Expr -> a
-- folde f _ (Val x) = f x
-- folde f g (Add a b) = g (folde f g a) (folde f g b)

-- -- 7
-- eval :: Expr -> Int
-- eval = folde (\x -> x) (+)

-- size :: Expr -> Int
-- size = folde (\_ -> 1) (+)

-- -- 8
-- data List a = Nil | a :- List a
-- infixr 5 :-

-- data Sem = Green | Yellow | Red
--   deriving (Eq, Show)

-- count :: Sem -> List Sem -> Int
-- count _ Nil = 0
-- count x (y :- ys) | x == y = 1 + count x ys
--                   | otherwise = count x ys

-- next :: Sem -> Sem
-- next Green = Yellow
-- next Yellow = Red
-- next Red = Green

-- -- 9
-- updateSems :: List Sem -> List Sem
-- updateSems Nil = Nil
-- updateSems (x:-xs) = next x :- updateSems xs

-- timeList :: List Sem -> Int
-- timeList Nil = 0
-- timeList (x:-xs) = case x of
--             Red -> 2 + timeList (updateSems xs)
--             Green -> 1 + timeList (updateSems xs)
--             Yellow -> 1 + timeList (updateSems xs)

-- -- 10
-- redl :: (b -> a -> b) -> b -> List a -> b
-- redl _ b Nil = b
-- redl f b (l:-ls) = redl f (f b l) ls

-- -- 11
-- -- timeList2 :: List Sem -> Int
-- -- timeList2 Nil = 0
-- -- timeList2 xs = fst $ redl calc (0, id) xs
-- --           where
-- --               calc (b, f) sem
-- --                 | f sem == Red = (b + 2, (next . f))
-- --                 | otherwise = (b + 1, (next . f))

-- -- timeListRedl :: List Sem -> Int
-- -- timeListRedl xs = fst $ redl foldFun (0, id) xs
-- --   where
-- --     foldFun (n, f) c
-- --       | f c == Red = (n + 2, next . next . f)
-- --       | otherwise = (n + 1, next . f)

-- -- quiz manhã

-- data Nota = Do | Re | Mi | Fa | Sol | La | Si
--   deriving (Show, Eq)

-- data ModoGrego = Jônio | Dórico | Frígio | Lídio | Mixolídio | Eólio | Lócrio
--   deriving (Show, Eq)

-- modoParaNotas :: ModoGrego -> [Nota]
-- modoParaNotas Jônio     = [Do, Re, Mi, Fa, Sol, La, Si]
-- modoParaNotas Dórico    = [Re, Mi, Fa, Sol, La, Si, Do]
-- modoParaNotas Frígio    = [Mi, Fa, Sol, La, Si, Do, Re]
-- modoParaNotas Lídio     = [Fa, Sol, La, Si, Do, Re, Mi]
-- modoParaNotas Mixolídio = [Sol, La, Si, Do, Re, Mi, Fa]
-- modoParaNotas Eólio     = [La, Si, Do, Re, Mi, Fa, Sol]
-- modoParaNotas Lócrio    = [Si, Do, Re, Mi, Fa, Sol, La]

-- gerarModo :: Int -> ModoGrego -> [Nota]
-- gerarModo n m | n <= 0 = []
--               | otherwise = take n (modoParaNotas m) ++ gerarModo (n - 7) m

-- -- lista 4

-- -- lista 5

-- -- 1

-- data Resultado = Pontuacao Int | Cola
--   deriving Show

-- instance Semigroup Resultado where
--   Pontuacao x <> Pontuacao y = Pontuacao (x + y)
--   _ <> _                     = Cola

-- instance Monoid Resultado where
--   mempty = Pontuacao 0

-- -- 2
-- data Set a = Set [a]
--   deriving Eq

-- instance Show a => Show (Set a) where
--   show (Set xs) = "{" <> intercalate "," (fmap show xs) <> "}"

-- fromList :: Ord a => [a] -> Set a
-- fromList = Set . sort . nub

-- member :: Ord a => a -> Set a -> Bool
-- member element (Set xs) = elem element xs


-- insert :: Ord a => a -> Set a -> Set a
-- insert element (Set xs) = fromList (element:xs)

-- delete' :: Ord a => a -> Set a -> Set a
-- delete' element (Set xs) = fromList (delete element xs)

-- --3
-- -- data Set a = Set [a]
-- --   deriving Eq

-- instance Ord a => Semigroup (Set a) where
--   (Set x) <> (Set y) = fromList (x<>y)

-- instance Ord a => Monoid (Set a) where
--   mempty = Set []

-- --4
-- data Dieta = Vegano | Vegetariano | Tradicional
-- data Lanche = Lanche (Set String) Int Dieta

-- instance Semigroup Dieta where
--   Vegano <> Vegetariano      = Vegetariano
--   Vegetariano <> Vegano      = Vegetariano
--   Tradicional <> _           = Tradicional
--   _ <> Tradicional           = Tradicional
--   Vegano <> Vegano           = Vegano
--   Vegetariano <> Vegetariano = Vegetariano

-- instance Monoid Dieta where
--   mempty = Vegano

-- --5
-- instance Semigroup Lanche where
--   (Lanche ingrX precoX dietaX) <> (Lanche ingrY precoY dietaY) = Lanche (ingrX <> ingrY) (precoX + precoY) (dietaX <> dietaY)

-- instance Monoid Lanche where
--   mempty = Lanche mempty 0 mempty

-- -- 6
-- data Treee a = Leafe a | Nodee (Treee a) a (Treee a)
--   deriving Show

-- instance Functor Treee where
--   fmap f (Leafe a) = Leafe (f a)
--   fmap f (Nodee left x right) = Nodee (fmap f left) (f x) (fmap f right)

-- -- 7
-- arvorePossui :: Eq a => a -> Treee a -> Bool
-- arvorePossui x (Leafe y) = x == y
-- arvorePossui x (Nodee l y r) = x == y || arvorePossui x l || arvorePossui x r

-- -- 8

-- contaLetras :: Treee String -> Treee Int
-- contaLetras = fmap length

-- -- 9

-- instance Foldable Treee where
--   foldMap t (Leafe x) = t x
--   foldMap t (Nodee l x r) = foldMap t l <> t x <> foldMap t r

-- -- 10
-- convertString2Int :: String -> Maybe Int
-- convertString2Int = readMaybe

-- -- 11
-- nothingToZero :: Maybe Int -> Int
-- nothingToZero Nothing  =  0
-- nothingToZero (Just x) = x

-- -- 12
-- frutasDaArvore :: Treee String -> Int
-- frutasDaArvore t = getSum $ foldMap (Sum . nothingToZero . convertString2Int) t

-- 13
newtype ZipList a = Z [a]
  deriving Show

instance Functor ZipList where
fmap g (Z xs) = Z (fmapDefault g xs)

instance Applicative ZipList where
-- pure :: a -> ZipList a
pure x = Z (repeat x)

-- 15
newtype Identity' a = Identity' a
  deriving Show
data Pair a = Pair a a

-- instance Functor Identity' where
--   fmap g (Identity' a) = Identity' (g a)

-- instance Applicative Identity' where
--   pure = Identity'
--   Identity' g <*> x  = fmap g x

instance Functor Pair where
  fmap g (Pair x y) = Pair (g x) (g y)

instance Applicative Pair where
  pure x = Pair x x
  (Pair g h) <*> (Pair x y) = Pair (g x) (h y)

-- 16

data RLE a = Repeat Int a (RLE a) | End
  deriving (Eq, Show)

rleCons :: Eq a => a -> RLE a -> RLE a
rleCons x End = Repeat 1 x End
rleConse x (Repeat i y seq) = if (x == y) then Repeat (i+1) x seq else Repeat 1 x (Repeat i y seq)

-- 17

instance Foldable RLE where
  foldMap _ End = mempty
  foldMap g (Repeat i x xs) = foldMap g (replicate i x) <> foldMap g xs

main :: IO ()
main = do
  let x = Identity' "1"
      y = Identity' "2"

  print x