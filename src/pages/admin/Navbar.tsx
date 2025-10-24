import { Box, Button, CloseButton, Drawer, Flex, Heading, IconButton, Input, InputGroup, Menu, Portal, Text, useDisclosure } from '@chakra-ui/react';
import React from 'react';
import { IoMenu } from 'react-icons/io5';
import { LuSearch } from 'react-icons/lu';
import { MdNotificationsNone } from 'react-icons/md';

const Navbar: React.FC = () => {
    return (
        <Flex alignItems={"center"} bg="#fff" h="100%" w={"100%"} p={4} zIndex={111} gap={4} boxShadow="0 2px 4px rgba(0, 0, 0, 0.1)">
            <Flex display={{ base: "flex", md: "none" }}>

                <Drawer.Root>
                    <Drawer.Trigger asChild>
                        <IconButton variant="outline" size="sm"><IoMenu size={20} /></IconButton>
                    </Drawer.Trigger>
                    <Portal>
                        <Drawer.Backdrop />
                        <Drawer.Positioner>
                            <Drawer.Content>
                                <Drawer.Header>
                                    <Drawer.Title>Drawer Title</Drawer.Title>
                                </Drawer.Header>
                                <Drawer.Body>
                                    <p>
                                        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do
                                        eiusmod tempor incididunt ut labore et dolore magna aliqua.
                                    </p>
                                </Drawer.Body>
                                <Drawer.Footer>
                                    <Button variant="outline">Cancel</Button>
                                    <Button>Save</Button>
                                </Drawer.Footer>
                                <Drawer.CloseTrigger asChild>
                                    <CloseButton size="sm" />
                                </Drawer.CloseTrigger>
                            </Drawer.Content>
                        </Drawer.Positioner>
                    </Portal>
                </Drawer.Root>
            </Flex>

            <Box w="270px">
                <Heading>Logo</Heading>
            </Box>

            <Flex alignItems="center" w={'100%'} justifyContent="space-between" gap="20px">
                <Flex alignItems="center" display={{ base: "none", lg: "block" }}>
                    <Text fontSize="20px" fontWeight="semibold">
                        {"Dashboard"}
                    </Text>
                </Flex>

                <InputGroup w={'300px'} startElement={<LuSearch />}>
                    <Input placeholder="Search anything" />
                </InputGroup>

                <Flex gap={4} alignItems={'center'}>
                    <Menu.Root>
                        <Menu.Trigger asChild>
                            <IconButton variant="outline" borderRadius={'full'} size="sm">
                                <MdNotificationsNone />
                            </IconButton>
                        </Menu.Trigger>
                        <Portal>
                            <Menu.Positioner>
                                <Menu.Content>
                                    <Menu.Item value="rename">Profile</Menu.Item>
                                    <Menu.Item value="delete" color="fg.error" _hover={{ bg: "bg.error", color: "fg.error" }}>
                                        Logout...
                                    </Menu.Item>
                                </Menu.Content>
                            </Menu.Positioner>
                        </Portal>
                    </Menu.Root>
                </Flex>
            </Flex>
        </Flex>
    );
};

export default Navbar;