{pkgs}: {
  deps = [
    pkgs.libuv
    pkgs.zlib
    pkgs.c-ares
    pkgs.util-linux
    pkgs.cacert
    pkgs.pkg-config
    pkgs.openssl
    pkgs.grpc
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.re2
    pkgs.oneDNN
    pkgs.glibcLocales
    pkgs.openssh
    pkgs.libxcrypt
    pkgs.bash
  ];
}
