import useFetch from '@/hooks/useFetch';
import { Box, Button, Field, Fieldset, Flex, Heading, Image, Input, Link, Stack, Text } from '@chakra-ui/react';
import React from 'react';
import { useForm } from 'react-hook-form';
import { useNavigate } from 'react-router';

const Login: React.FC = () => {

    const { register, handleSubmit, formState: { errors } } = useForm();
    // const { setUsername } = useContext(AppContext);
    const navigate = useNavigate();
    // const toast = useToast();

    const { request, loading } = useFetch(); // use custom hook

    const onSubmit = async (data: any) => {
        try {
            const response = await request({
                url: "/user/login",
                method: "POST",
                body: {
                    email: data.email,
                    password: data.password,
                },
            });

            if (!response || response.error) {
                // toast({
                //     title: "Login Failed",
                //     description: response?.error || "Something went wrong",
                //     status: "error",
                //     duration: 5000,
                //     position: "top",
                //     isClosable: true,
                // });
                console.error(response?.error);
                return;
            }

            const userData = response.data;
            // setUsername(userData.name);

            // store token and encrypted user
            localStorage.setItem("token", response.auth_token);
            // localStorage.setItem("chatLimitReached", "false");
            // const encryptedData = await encrypt(userData);
            localStorage.setItem("user", userData);

            // toast({
            //     title: "Login Successful!",
            //     description: "Welcome back!",
            //     status: "success",
            //     duration: 5000,
            //     position: "top",
            //     isClosable: true,
            // });

            navigate(`/${userData.id}`);
        } catch (err: any) {
            // toast({
            //     title: "Login Failed",
            //     description: err?.message || "Something went wrong",
            //     status: "error",
            //     duration: 5000,
            //     position: "top",
            //     isClosable: true,
            // });
        }
    };
    return (
        <>
            <Box bg="#fff" minH="100vh">
                <Flex justify="center" align="center" h="100vh" position="relative">
                    {/* Logo in top-left corner */}
                    <Flex position="absolute" top="20px" left="30px">
                        {/* <Image boxSize="40px" src={Logo} alt="Logo" /> */}
                    </Flex>

                    {/* Login Card */}
                    <Flex
                        maxW="400px"
                        w="full"
                        bg={"gray.100"}
                        color={"black"}
                        justify="center"
                        align="center"
                        direction="column"
                        borderRadius="lg"
                        border={`1px solid grey`}
                        boxShadow="0 0 4px #cbcbcb94, 0 0 8px #cbcbcb3b"
                        py={8}
                        px={6}
                    >
                        <Heading mb={6} fontSize="2xl">
                            Login
                        </Heading>

                        <form onSubmit={handleSubmit(onSubmit)}>
                            <Fieldset.Root w="100%">
                                <Stack >
                                    <Fieldset.Legend>Login</Fieldset.Legend>
                                </Stack>

                                <Fieldset.Content w={"300px"}>
                                    {/* Email Field */}
                                    <Field.Root >
                                        <Field.Label>Email</Field.Label>
                                        <Input
                                            placeholder="eg: john@example.com"
                                            type="email"
                                            {...register("email", {
                                                required: "Email is required",
                                                pattern: {
                                                    value:
                                                        /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/,
                                                    message: "Invalid email format",
                                                },
                                            })}
                                        />
                                        {/* {errors.email && (
                                            <Field.RequiredIndicator>{errors.email.message}</Field.RequiredIndicator>
                                        )} */}
                                    </Field.Root>

                                    {/* Password Field */}
                                    <Field.Root >
                                        <Field.Label>Password</Field.Label>
                                        <Input
                                            placeholder="Enter your password"
                                            type="password"
                                            {...register("password", {
                                                required: "Password is required",
                                            })}
                                        />
                                        {/* {errors.password && (
                                            <Field.ErrorMessage>{errors.password.message}</Field.ErrorMessage>
                                        )} */}
                                    </Field.Root>

                                    <Button
                                        type="submit"
                                        mt={4}
                                        width="100%"
                                        bg="#ED3438"
                                        color="white"
                                        _hover={{ bg: "#d62f33" }}
                                    >
                                        Login
                                    </Button>
                                </Fieldset.Content>
                            </Fieldset.Root>
                        </form>
                        {/* <Text fontSize="sm" mt={3} color="gray.500">
                            Don&apos;t have an account?{" "}
                            <Link to="/signup" style={{ color: "#0082ff" }}>
                                Sign Up
                            </Link>
                        </Text> */}
                    </Flex>
                </Flex>
            </Box>
        </>
    );
};

export default Login;