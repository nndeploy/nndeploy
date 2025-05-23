import styled from 'styled-components';

export const Header = styled.div`
  box-sizing: border-box;
  display: flex;
  justify-content: flex-start;
  align-items: center;
  width: 100%;
  column-gap: 8px;
  border-radius: 8px 8px 0 0;
  cursor: move;

  background: linear-gradient(#f2f2ff 0%, rgba(0, 0, 0, 0.02) 100%);
  overflow: hidden;

  padding: 8px;
`;

export const Title = styled.div`
  font-size: 20px;
  flex: 1;
  width: 0;
`;

export const Icon = styled.img`
  width: 24px;
  height: 24px;
  scale: 0.8;
  border-radius: 4px;
`;

export const Operators = styled.div`
  display: flex;
  align-items: center;
  column-gap: 4px;
`;
