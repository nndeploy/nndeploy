import { IconExit, IconFile, IconGithubLogo, IconHelpCircle, IconUser } from "@douyinfe/semi-icons"
import { Avatar, Button, Dropdown, Nav } from "@douyinfe/semi-ui"
import { faBilibili, faDeploydog, faDiscord, faWeixin, faZhihu } from "@fortawesome/free-brands-svg-icons"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import companyLogo from "../../../../assets/kapybara_logo.png";
import { useNavigate } from "react-router-dom";
import './index.scss'
import { IconTabs } from "@douyinfe/semi-icons-lab";

const Header: React.FC = () => {

  const navigate = useNavigate();
  return (
    <Nav mode="horizontal" className="topNav" >
      <Nav.Header>
        <img
          src={companyLogo}
          width="100"
          alt="Logo"
          className="companyLogo"
          onClick={() => {
            navigate('/')
          }}
          title="home"
        />
      </Nav.Header>
      {/* <Dropdown
        trigger="hover"
        render={
          <Dropdown.Menu>
            <Dropdown.Item>
              <a href="https://github.com/nndeploy/nndeploy" target="_blank">
                <Button icon={<IconGithubLogo />} theme="borderless" size='large' />github
              </a></Dropdown.Item>
            <Dropdown.Item>
              <a
                href="https://nndeploy-zh.readthedocs.io/zh-cn/latest/"
                target="_blank"
              >
                <Button icon={<IconHelpCircle />} size='large' theme="borderless" />
              </a></Dropdown.Item>
          </Dropdown.Menu>
        }
      >
        <Button theme="borderless"
          //</Dropdown>icon={<IconHelpCircle />}

          style={{ marginRight: 8 }}>
          help
        </Button>
      </Dropdown> */}

      <Nav.Footer >
        {/* <a href="https://github.com/nndeploy/nndeploy" target="_blank">
          <Button icon={<IconGithubLogo />} theme="borderless" size='large' />
        </a>

        <a
          href="https://nndeploy-zh.readthedocs.io/zh-cn/latest/"
          target="_blank"
        >
          <Button icon={<IconHelpCircle />} size='large' theme="borderless" />
        </a>
        <a
          href="https://www.zhihu.com/column/c_1690464325314240512"
          target="_blank"
        >
          <Button
            icon={<FontAwesomeIcon icon={faZhihu} size="1x" />}
            size='large'
            theme="borderless"
          />
        </a>
        <a href="https://discord.gg/9rUwfAaMbr" target="_blank">
          <Button
            icon={<FontAwesomeIcon icon={faDiscord} size="1x" />}
            size='large'
            theme="borderless"
          />
        </a>
        <a href="https://www.bilibili.com/video/BV1HU7CznE39/?spm_id_from=333.1387.collection.video_card.click&vd_source=c5d7760172919cd367c00bf4e88d6f57"
          target="_blank">
          <Button
            icon={<FontAwesomeIcon icon={faBilibili} size="1x" />}
            size='large'
            theme="borderless"
          />
        </a>
        <a
          href="https://github.com/nndeploy/nndeploy/blob/main/docs/zh_cn/knowledge_shared/wechat.md"
          target="_blank"
        >
          <Button
            icon={<FontAwesomeIcon icon={faWeixin} size="1x" />}
            theme="borderless"
            size='large'
          />
          
        </a> */}
        <Dropdown
          //position="bottomLeft"

          render={
            <Dropdown.Menu >
              <Dropdown.Item>
                <a href="https://github.com/nndeploy/nndeploy" target="_blank" className="helpItem">
                  <IconGithubLogo /> Github Issues
                </a></Dropdown.Item>
              <Dropdown.Item>
                <a href="https://github.com/nndeploy/nndeploy" target="_blank" className="helpItem">

                  <IconFile /> View Docs
                </a>
              </Dropdown.Item>
              {/* <Dropdown.Item>
                <a href="https://www.zhihu.com/column/c_1690464325314240512" target="_blank" className="helpItem">

                  <FontAwesomeIcon icon={faZhihu} size="1x" /> zhihu
                </a>
              </Dropdown.Item> */}
              <Dropdown.Item>
                <a href="https://discord.gg/9rUwfAaMbr" target="_blank" className="helpItem">

                  <FontAwesomeIcon icon={faDiscord} size="1x" /> Discord Group
                </a>
              </Dropdown.Item>
              {/* <Dropdown.Item>
                <a href="https://discord.spm_id_from/9rUwfAaMbr" target="_blank" className="helpItem">

                  <FontAwesomeIcon icon={faBilibili} size="1x" /> bilibili
                </a>
              </Dropdown.Item> */}
              <Dropdown position={'leftTop'}
                render={
                  <Dropdown.Menu>
                    <Dropdown.Item>
                      <img
                        src="https://github.com/nndeploy/nndeploy/raw/main/docs/image/wechat.jpg"
                        alt="wechat"
                        width="200"
                      />
                    </Dropdown.Item>
                  </Dropdown.Menu>

                }>
                <Dropdown.Item>
                  <a href="#" target="_blank" className="helpItem">
                    <FontAwesomeIcon icon={faWeixin} size="1x" /> 
                     Wechat Group
                   
                  </a>
                </Dropdown.Item>
              </Dropdown>

               <Dropdown.Item>
                <a href="https://github.com/nndeploy/nndeploy/blob/main/docs/zh_cn/quick_start/deploy.md" target="_blank" className="helpItem">

                  <IconExit /> 
                  {/* <FontAwesomeIcon icon={faDeploydog} size="1x" />  */}
                  Deployment 
                </a>
              </Dropdown.Item>

            </Dropdown.Menu>
          }
        >
         {/* <Avatar size="small"  color="blue" style={{marginRight: '10px', cursor: 'pointer'}} ><IconHelpCircle /></Avatar>  */}
         <Button icon={<IconHelpCircle />} size='large' theme="borderless" />
          {/* <Nav.Item itemKey="home" text={
            <a href="https://github.com/nndeploy/nndeploy" target="_blank" className="helpItem">

              <IconGithubLogo /> github
            </a>
          }

          />
          <Nav.Item itemKey="docs" text={
            <a href="https://github.com/nndeploy/nndeploy" target="_blank" className="helpItem">

              <IconHelpCircle /> docs
            </a>
          }
          />
          <Nav.Item itemKey="zhihu" text={
            <a href="https://www.zhihu.com/column/c_1690464325314240512" target="_blank" className="helpItem">

              <FontAwesomeIcon icon={faZhihu} size="1x" /> zhihu
            </a>
          }

          />
          <Nav.Item itemKey="discord" text={
            <a href="https://discord.gg/9rUwfAaMbr" target="_blank" className="helpItem">

              <FontAwesomeIcon icon={faDiscord} size="1x" /> discord
            </a>
          }
          />
          <Nav.Item itemKey="bilibili" text={
            <a href="https://discord.spm_id_from/9rUwfAaMbr" target="_blank" className="helpItem">

              <FontAwesomeIcon icon={faBilibili} size="1x" /> bilibili
            </a>
          }
          />

          <Nav.Sub itemKey="wechat" text={
            <a href="https://github.com/nndeploy/nndeploy/raw/main/docs/image/wechat.jpg" target="_blank"
              className="helpItem"
            >

              <FontAwesomeIcon icon={faWeixin} size="1x" /> wechat
            </a>
          }>
            <Nav.Item itemKey="wechat-image" text={
              <img
                src="https://github.com/nndeploy/nndeploy/raw/main/docs/image/wechat.jpg"
                alt="wechat"
                width="200"
              />
            } />
          </Nav.Sub> */}

        </Dropdown>

        <Dropdown
         // position="bottomLeft"
          render={
            <Dropdown.Menu>
              <Dropdown.Item>Profile</Dropdown.Item>
              <Dropdown.Item>Logout</Dropdown.Item>
            </Dropdown.Menu>
          }
        >
          <Avatar size="small" color="blue">
            <IconUser size="small" />
          </Avatar>
        </Dropdown>
      </Nav.Footer>
    </Nav >
  )
}

export default Header;
