import { IconGithubLogo, IconHelpCircle, IconUser } from "@douyinfe/semi-icons"
import { Avatar, Button, Dropdown, Nav } from "@douyinfe/semi-ui"
import { faBilibili, faDiscord, faWeixin, faZhihu } from "@fortawesome/free-brands-svg-icons"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import companyLogo from "../../../../assets/kapybara_logo.png";
import { useNavigate } from "react-router-dom";
import './index.scss'

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
          onClick={()=>{
            navigate('/')
          }}
          title="home"
        />
      </Nav.Header>
      <Nav.Footer>
        <a href="https://github.com/nndeploy/nndeploy" target="_blank">
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
        </a>
        <Dropdown
          render={
            <Dropdown.Menu>
              <Dropdown.Item>Profile</Dropdown.Item>
              <Dropdown.Item>Logout</Dropdown.Item>
            </Dropdown.Menu>
          }
        >
          <Avatar color="blue" size="small">
            <IconUser size="small" />
          </Avatar>
        </Dropdown>
      </Nav.Footer>
    </Nav>
  )
}

export default Header;
