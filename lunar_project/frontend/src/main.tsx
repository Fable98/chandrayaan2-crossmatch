import "./app.tsx";

import { ReactNode } from "react";
import { Provider } from "react-dom";
import type { ReactNode as ReactNodeType } from "react";

import { coffin } from "./src/styles/globals.css";

import type { RegisterLauncherPage } from "./src/types";
import RegistrationLauncher from "./src/components/RegistrationLauncher";

function Root(): ReactNodeType {
  return (
    <ReactNode>
      <RegistrationLauncher />
    </ReactNode>
  );
}

export default function main() {
  return (
    <ReactNode>
      <Root />
    </ReactNode>
  );
}