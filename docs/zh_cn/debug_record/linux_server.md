
# 无法git clone，具体报错fatal: unable to access 'https://github.com/wnlen/clash-for-linux.git/': gnutls_handshake() failed: Error in the pull function.(开发者无法git clone，无法在服务器上做开发)


不安装代理，git clone失败；
安装了代理，依旧失败（安装方式如下：ubuntu服务器(无gui)安装clash，配置代理，使用[clash-for-linux](https://github.com/wnlen/clash-for-linux)，流程显示成功，但是依旧无法连接外网）

## 订阅链接没问题
- 在windows上使用clash-for-windows，使用该订阅链接，连接成功，可以连接外网
- 该订阅链接网站显示一台机器链接成功

## 7890端口未被占用
- http_proxy为127.0.0.1:7890，https_proxy为127.0.0.1:7890，no_proxy为127.0.0.1,localhost

## 机器没有防火墙

## 当前现象

- ping 127.0.0.1 成功
- www.baidu.com 
  - ping/wget/curl 成功
- www.github.com
  - ping成功
  - wget失败
    - 错误信息：--2025-01-18 12:53:35--  http://www.github.com/
Connecting to 127.0.0.1:7890... connected.
Proxy request sent, awaiting response... 502 Bad Gateway
2025-01-18 12:53:40 ERROR 502: Bad Gateway
  - curl失败，没有任何打印信息
- ssh -T git@github.com 成功
  - Hi Alwaysssssss! You've successfully authenticated, but GitHub does not provide shell access.
- www.google.com
  - PING www.google.com (31.13.88.169) 56(84) bytes of data.卡死
  - wget失败
  - curl失败，没有任何打印信息


# 解决办法，基于ssh的git clone（git clone --recursive git@github.com:nndeploy/nndeploy.git）（参照realtyxxx的配置）

- 在根目录中`$HOME/.ssh/config`中增加如下代码

```bash
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
```

- 在根目录中`$HOME/.gitconfig`中增加如下代码
```bash
[user]
  # 修改为用户邮箱
	email = user@gmail.com 
  # 修改为开发者用户名
	name = user 
[http]
	# version = HTTP/1.1
	sslVerify = true
	# sslBackend = openssl
[https]
	sslVerify = true
[url "ssh://git@github.com/"]
	insteadOf = https://github.com/
[credential]
	helper = manager-core
```

