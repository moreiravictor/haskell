{-# LANGUAGE CPP #-}
{-# LANGUAGE NoRebindableSyntax #-}
#if __GLASGOW_HASKELL__ >= 810
{-# OPTIONS_GHC -Wno-prepositive-qualified-module #-}
#endif
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
{-# OPTIONS_GHC -w #-}
module Paths_lista_dois (
    version,
    getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where


import qualified Control.Exception as Exception
import qualified Data.List as List
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude


#if defined(VERSION_base)

#if MIN_VERSION_base(4,0,0)
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#else
catchIO :: IO a -> (Exception.Exception -> IO a) -> IO a
#endif

#else
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#endif
catchIO = Exception.catch

version :: Version
version = Version [0,1,0,0] []

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir `joinFileName` name)

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath




bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath
bindir     = "/home/redrest/Documents/dev/personal/lista-dois/.stack-work/install/x86_64-linux/9991e07d9b63f63dc225b16d43fd6972f2a146a5e8a2e9b646f3798f22a27e82/9.6.6/bin"
libdir     = "/home/redrest/Documents/dev/personal/lista-dois/.stack-work/install/x86_64-linux/9991e07d9b63f63dc225b16d43fd6972f2a146a5e8a2e9b646f3798f22a27e82/9.6.6/lib/x86_64-linux-ghc-9.6.6/lista-dois-0.1.0.0-ISdmbp4vwWizYGJe4pIRf-lista-dois"
dynlibdir  = "/home/redrest/Documents/dev/personal/lista-dois/.stack-work/install/x86_64-linux/9991e07d9b63f63dc225b16d43fd6972f2a146a5e8a2e9b646f3798f22a27e82/9.6.6/lib/x86_64-linux-ghc-9.6.6"
datadir    = "/home/redrest/Documents/dev/personal/lista-dois/.stack-work/install/x86_64-linux/9991e07d9b63f63dc225b16d43fd6972f2a146a5e8a2e9b646f3798f22a27e82/9.6.6/share/x86_64-linux-ghc-9.6.6/lista-dois-0.1.0.0"
libexecdir = "/home/redrest/Documents/dev/personal/lista-dois/.stack-work/install/x86_64-linux/9991e07d9b63f63dc225b16d43fd6972f2a146a5e8a2e9b646f3798f22a27e82/9.6.6/libexec/x86_64-linux-ghc-9.6.6/lista-dois-0.1.0.0"
sysconfdir = "/home/redrest/Documents/dev/personal/lista-dois/.stack-work/install/x86_64-linux/9991e07d9b63f63dc225b16d43fd6972f2a146a5e8a2e9b646f3798f22a27e82/9.6.6/etc"

getBinDir     = catchIO (getEnv "lista_dois_bindir")     (\_ -> return bindir)
getLibDir     = catchIO (getEnv "lista_dois_libdir")     (\_ -> return libdir)
getDynLibDir  = catchIO (getEnv "lista_dois_dynlibdir")  (\_ -> return dynlibdir)
getDataDir    = catchIO (getEnv "lista_dois_datadir")    (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "lista_dois_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "lista_dois_sysconfdir") (\_ -> return sysconfdir)



joinFileName :: String -> String -> FilePath
joinFileName ""  fname = fname
joinFileName "." fname = fname
joinFileName dir ""    = dir
joinFileName dir fname
  | isPathSeparator (List.last dir) = dir ++ fname
  | otherwise                       = dir ++ pathSeparator : fname

pathSeparator :: Char
pathSeparator = '/'

isPathSeparator :: Char -> Bool
isPathSeparator c = c == '/'
